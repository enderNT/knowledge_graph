package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

type fakeAnthropicClient struct {
	content map[string]any
	err     error
	req     generateJSONRequest
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f *fakeAnthropicClient) GenerateJSON(_ context.Context, req generateJSONRequest) (map[string]any, error) {
	f.req = req
	if f.err != nil {
		return nil, f.err
	}
	return f.content, nil
}

func (fn roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return fn(req)
}

func TestHandleGenerateJSONRequiresBearerToken(t *testing.T) {
	server := gatewayServer{
		cfg: config{active: true, bearerToken: "secret"},
		client: &fakeAnthropicClient{
			content: map[string]any{"ok": true},
		},
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/generate-json", strings.NewReader(`{"system_prompt":"x","user_payload_json":{"a":1}}`))
	rec := httptest.NewRecorder()

	server.handleGenerateJSON(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401, got %d", rec.Code)
	}
}

func TestHandleGenerateJSONReturnsStructuredJSON(t *testing.T) {
	client := &fakeAnthropicClient{
		content: map[string]any{"domain": "Historia"},
	}
	server := gatewayServer{
		cfg:    config{active: true, bearerToken: "secret"},
		client: client,
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/generate-json", strings.NewReader(`{"system_prompt":"x","user_payload_json":{"a":1}}`))
	req.Header.Set("Authorization", "Bearer secret")
	rec := httptest.NewRecorder()

	server.handleGenerateJSON(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	var payload map[string]any
	if err := json.NewDecoder(rec.Body).Decode(&payload); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	content, ok := payload["content_json"].(map[string]any)
	if !ok || content["domain"] != "Historia" {
		t.Fatalf("unexpected content_json: %#v", payload["content_json"])
	}
	if client.req.SystemPrompt != "x" {
		t.Fatalf("expected forwarded system prompt, got %q", client.req.SystemPrompt)
	}
}

func TestHandleGenerateJSONValidatesThinkingModes(t *testing.T) {
	server := gatewayServer{
		cfg: config{active: true, bearerToken: "secret"},
		client: &fakeAnthropicClient{
			content: map[string]any{"ok": true},
		},
	}

	testCases := []struct {
		name string
		body string
	}{
		{
			name: "adaptive with budget",
			body: `{"system_prompt":"x","user_payload_json":{"a":1},"thinking":{"type":"adaptive","budget_tokens":1000}}`,
		},
		{
			name: "enabled without budget",
			body: `{"system_prompt":"x","user_payload_json":{"a":1},"thinking":{"type":"enabled"}}`,
		},
		{
			name: "invalid effort",
			body: `{"system_prompt":"x","user_payload_json":{"a":1},"output_config":{"effort":"turbo"}}`,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/v1/generate-json", strings.NewReader(tc.body))
			req.Header.Set("Authorization", "Bearer secret")
			rec := httptest.NewRecorder()

			server.handleGenerateJSON(rec, req)

			if rec.Code != http.StatusBadRequest {
				t.Fatalf("expected 400, got %d", rec.Code)
			}
		})
	}
}

func TestHandleGenerateJSONMapsClientErrors(t *testing.T) {
	testCases := []struct {
		name       string
		clientErr  error
		statusCode int
	}{
		{name: "upstream auth", clientErr: errUpstreamAuth, statusCode: http.StatusBadGateway},
		{name: "timeout", clientErr: errUpstreamTimeout, statusCode: http.StatusGatewayTimeout},
		{name: "invalid json", clientErr: errUpstreamJSON, statusCode: http.StatusBadGateway},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			server := gatewayServer{
				cfg: config{active: true, bearerToken: "secret"},
				client: &fakeAnthropicClient{
					err: tc.clientErr,
				},
			}

			req := httptest.NewRequest(http.MethodPost, "/v1/generate-json", strings.NewReader(`{"system_prompt":"x","user_payload_json":{"a":1}}`))
			req.Header.Set("Authorization", "Bearer secret")
			rec := httptest.NewRecorder()

			server.handleGenerateJSON(rec, req)

			if rec.Code != tc.statusCode {
				t.Fatalf("expected %d, got %d", tc.statusCode, rec.Code)
			}
		})
	}
}

func TestHTTPAnthropicClientGenerateJSONParsesResponse(t *testing.T) {
	client := &httpAnthropicClient{
		baseURL:      "http://anthropic.test",
		apiKey:       "sk-test",
		defaultModel: "claude-test",
		httpClient: &http.Client{
			Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
				if got := r.Header.Get("X-API-Key"); got != "sk-test" {
					t.Fatalf("unexpected api key header: %q", got)
				}
				var body map[string]any
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Fatalf("decode request body: %v", err)
				}
				thinking, ok := body["thinking"].(map[string]any)
				if !ok || thinking["type"] != "adaptive" {
					t.Fatalf("unexpected thinking payload: %#v", body["thinking"])
				}
				outputConfig, ok := body["output_config"].(map[string]any)
				if !ok || outputConfig["effort"] != "medium" {
					t.Fatalf("unexpected output_config payload: %#v", body["output_config"])
				}
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     make(http.Header),
					Body:       io.NopCloser(strings.NewReader(`{"content":[{"type":"text","text":"{\"domain\":\"Historia\"}"}]}`)),
				}, nil
			}),
		},
	}

	content, err := client.GenerateJSON(context.Background(), generateJSONRequest{
		SystemPrompt:    "prompt",
		UserPayloadJSON: json.RawMessage(`{"text":"hola"}`),
		Thinking:        &thinkingConfig{Type: "adaptive"},
		OutputConfig:    &outputConfig{Effort: "medium"},
	})
	if err != nil {
		t.Fatalf("GenerateJSON returned error: %v", err)
	}
	if content["domain"] != "Historia" {
		t.Fatalf("unexpected content: %#v", content)
	}
}

func TestHTTPAnthropicClientGenerateJSONRejectsInvalidJSON(t *testing.T) {
	client := &httpAnthropicClient{
		baseURL:      "http://anthropic.test",
		apiKey:       "sk-test",
		defaultModel: "claude-test",
		httpClient: &http.Client{
			Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     make(http.Header),
					Body:       io.NopCloser(strings.NewReader(`{"content":[{"type":"text","text":"not-json"}]}`)),
				}, nil
			}),
		},
	}

	_, err := client.GenerateJSON(context.Background(), generateJSONRequest{
		SystemPrompt:    "prompt",
		UserPayloadJSON: json.RawMessage(`{"text":"hola"}`),
	})
	if !errors.Is(err, errUpstreamJSON) {
		t.Fatalf("expected errUpstreamJSON, got %v", err)
	}
}

func TestHTTPAnthropicClientGenerateJSONHandlesTimeout(t *testing.T) {
	client := &httpAnthropicClient{
		baseURL:      "http://anthropic.test",
		apiKey:       "sk-test",
		defaultModel: "claude-test",
		httpClient: &http.Client{
			Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
				return nil, context.DeadlineExceeded
			}),
		},
	}

	_, err := client.GenerateJSON(context.Background(), generateJSONRequest{
		SystemPrompt:    "prompt",
		UserPayloadJSON: json.RawMessage(`{"text":"hola"}`),
	})
	if !errors.Is(err, errUpstreamTimeout) {
		t.Fatalf("expected errUpstreamTimeout, got %v", err)
	}
}
