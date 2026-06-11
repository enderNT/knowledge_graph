package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

type config struct {
	port         string
	active       bool
	apiKey       string
	bearerToken  string
	defaultModel string
	baseURL      string
	timeout      time.Duration
}

type generateJSONRequest struct {
	Model           string          `json:"model"`
	SystemPrompt    string          `json:"system_prompt"`
	UserPayloadJSON json.RawMessage `json:"user_payload_json"`
	Temperature     *float64        `json:"temperature,omitempty"`
	Thinking        *thinkingConfig `json:"thinking,omitempty"`
	OutputConfig    *outputConfig   `json:"output_config,omitempty"`
}

type thinkingConfig struct {
	Type         string `json:"type"`
	BudgetTokens *int   `json:"budget_tokens,omitempty"`
}

type outputConfig struct {
	Effort string `json:"effort,omitempty"`
}

type anthropicClient interface {
	GenerateJSON(ctx context.Context, req generateJSONRequest) (map[string]any, error)
}

type gatewayServer struct {
	cfg    config
	client anthropicClient
	now    func() time.Time
}

type httpAnthropicClient struct {
	baseURL      string
	apiKey       string
	defaultModel string
	httpClient   *http.Client
}

var (
	errGatewayAuth     = errors.New("gateway_auth_failed")
	errGatewayDisabled = errors.New("anthropic_gateway_disabled")
	errUpstreamAuth    = errors.New("anthropic_auth_failed")
	errUpstreamTimeout = errors.New("anthropic_timeout")
	errUpstreamJSON    = errors.New("anthropic_json_invalid")
	errUpstreamGeneric = errors.New("anthropic_upstream_error")
)

func main() {
	cfg := loadConfig()

	if cfg.active {
		switch {
		case cfg.apiKey == "":
			log.Fatal("ANTHROPIC_API_KEY is required when AI_PROVIDER=anthropic")
		case cfg.bearerToken == "":
			log.Fatal("ANTHROPIC_GATEWAY_BEARER_TOKEN is required when AI_PROVIDER=anthropic")
		case cfg.defaultModel == "":
			log.Fatal("ANTHROPIC_CHAT_MODEL is required when AI_PROVIDER=anthropic")
		}
	}

	server := gatewayServer{
		cfg: cfg,
		client: &httpAnthropicClient{
			baseURL:      cfg.baseURL,
			apiKey:       cfg.apiKey,
			defaultModel: cfg.defaultModel,
			httpClient:   &http.Client{Timeout: cfg.timeout},
		},
		now: time.Now,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health/live", server.handleLive)
	mux.HandleFunc("/health/ready", server.handleReady)
	mux.HandleFunc("/v1/generate-json", server.handleGenerateJSON)

	httpServer := &http.Server{
		Addr:              ":" + cfg.port,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
	}

	log.Printf("anthropic gateway listening on :%s active=%t", cfg.port, cfg.active)
	if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
}

func loadConfig() config {
	timeoutSeconds := 180.0
	if raw := strings.TrimSpace(os.Getenv("ANTHROPIC_TIMEOUT_SECONDS")); raw != "" {
		if parsed, err := strconv.ParseFloat(raw, 64); err == nil && parsed > 0 {
			timeoutSeconds = parsed
		}
	}

	port := strings.TrimSpace(os.Getenv("ANTHROPIC_GATEWAY_PORT"))
	if port == "" {
		port = "8081"
	}

	baseURL := strings.TrimSpace(os.Getenv("ANTHROPIC_BASE_URL"))
	if baseURL == "" {
		baseURL = "https://api.anthropic.com/v1"
	}

	return config{
		port:         port,
		active:       strings.EqualFold(strings.TrimSpace(os.Getenv("AI_PROVIDER")), "anthropic"),
		apiKey:       strings.TrimSpace(os.Getenv("ANTHROPIC_API_KEY")),
		bearerToken:  strings.TrimSpace(os.Getenv("ANTHROPIC_GATEWAY_BEARER_TOKEN")),
		defaultModel: strings.TrimSpace(os.Getenv("ANTHROPIC_CHAT_MODEL")),
		baseURL:      strings.TrimRight(baseURL, "/"),
		timeout:      time.Duration(timeoutSeconds * float64(time.Second)),
	}
}

func (s gatewayServer) handleLive(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"status": "live"})
}

func (s gatewayServer) handleReady(w http.ResponseWriter, _ *http.Request) {
	mode := "active"
	if !s.cfg.active {
		mode = "disabled"
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "ready", "mode": mode})
}

func (s gatewayServer) handleGenerateJSON(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		writeError(w, http.StatusUnauthorized, errGatewayAuth)
		return
	}
	if !s.cfg.active {
		writeError(w, http.StatusServiceUnavailable, errGatewayDisabled)
		return
	}
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		writeError(w, http.StatusMethodNotAllowed, errors.New("method_not_allowed"))
		return
	}

	var req generateJSONRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, errors.New("invalid_request_json"))
		return
	}
	if strings.TrimSpace(req.SystemPrompt) == "" || len(req.UserPayloadJSON) == 0 || !json.Valid(req.UserPayloadJSON) {
		writeError(w, http.StatusBadRequest, errors.New("invalid_request_payload"))
		return
	}
	if err := validateGenerateJSONRequest(req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}

	content, err := s.client.GenerateJSON(r.Context(), req)
	if err != nil {
		switch {
		case errors.Is(err, errUpstreamAuth):
			writeError(w, http.StatusBadGateway, err)
		case errors.Is(err, errUpstreamTimeout):
			writeError(w, http.StatusGatewayTimeout, err)
		case errors.Is(err, errUpstreamJSON):
			writeError(w, http.StatusBadGateway, err)
		default:
			writeError(w, http.StatusBadGateway, errUpstreamGeneric)
		}
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{"content_json": content})
}

func (s gatewayServer) authorized(r *http.Request) bool {
	expected := s.cfg.bearerToken
	if expected == "" && !s.cfg.active {
		return true
	}
	auth := strings.TrimSpace(r.Header.Get("Authorization"))
	return auth == "Bearer "+expected
}

func (c *httpAnthropicClient) GenerateJSON(ctx context.Context, req generateJSONRequest) (map[string]any, error) {
	model := strings.TrimSpace(req.Model)
	if model == "" {
		model = c.defaultModel
	}

	body := map[string]any{
		"model":      model,
		"max_tokens": 8192,
		"system":     req.SystemPrompt + " Return only valid JSON.",
		"messages": []map[string]any{
			{
				"role": "user",
				"content": []map[string]string{
					{
						"type": "text",
						"text": string(req.UserPayloadJSON),
					},
				},
			},
		},
	}
	if req.Temperature != nil && req.Thinking == nil {
		body["temperature"] = *req.Temperature
	}
	if req.Thinking != nil {
		thinking := map[string]any{
			"type": req.Thinking.Type,
		}
		if req.Thinking.BudgetTokens != nil {
			thinking["budget_tokens"] = *req.Thinking.BudgetTokens
		}
		body["thinking"] = thinking
	}
	if req.OutputConfig != nil && strings.TrimSpace(req.OutputConfig.Effort) != "" {
		body["output_config"] = map[string]any{"effort": req.OutputConfig.Effort}
	}

	rawBody, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/messages", bytes.NewReader(rawBody))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("X-API-Key", c.apiKey)
	httpReq.Header.Set("Anthropic-Version", "2023-06-01")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			return nil, errUpstreamTimeout
		}
		var netErr interface{ Timeout() bool }
		if errors.As(err, &netErr) && netErr.Timeout() {
			return nil, errUpstreamTimeout
		}
		return nil, errUpstreamGeneric
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		return nil, errUpstreamAuth
	}
	if resp.StatusCode >= 400 {
		var errBody bytes.Buffer
		_, _ = errBody.ReadFrom(resp.Body)
		log.Printf("anthropic upstream error: status=%d body=%s", resp.StatusCode, errBody.String())
		return nil, errUpstreamGeneric
	}

	var payload struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, errUpstreamGeneric
	}

	var parts []string
	for _, block := range payload.Content {
		if block.Type == "text" && strings.TrimSpace(block.Text) != "" {
			parts = append(parts, block.Text)
		}
	}
	if len(parts) == 0 {
		return nil, errUpstreamJSON
	}

	joined := strings.TrimSpace(strings.Join(parts, "\n"))
	start := strings.Index(joined, "{")
	end := strings.LastIndex(joined, "}")
	if start < 0 || end < start {
		return nil, errUpstreamJSON
	}

	var content map[string]any
	if err := json.Unmarshal([]byte(joined[start:end+1]), &content); err != nil {
		return nil, errUpstreamJSON
	}
	return content, nil
}

func writeJSON(w http.ResponseWriter, status int, payload map[string]any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func writeError(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, map[string]any{"detail": err.Error()})
}

func validateGenerateJSONRequest(req generateJSONRequest) error {
	if req.Thinking != nil {
		thinkingType := strings.TrimSpace(req.Thinking.Type)
		switch thinkingType {
		case "adaptive":
			if req.Thinking.BudgetTokens != nil {
				return errors.New("thinking budget_tokens is not allowed when thinking.type=adaptive")
			}
		case "enabled":
			if req.Thinking.BudgetTokens == nil || *req.Thinking.BudgetTokens <= 0 {
				return errors.New("thinking budget_tokens must be greater than 0 when thinking.type=enabled")
			}
		default:
			return errors.New("thinking.type must be adaptive or enabled")
		}
	}

	if req.OutputConfig != nil {
		switch strings.TrimSpace(req.OutputConfig.Effort) {
		case "", "low", "medium", "high", "max", "xhigh":
		default:
			return errors.New("output_config.effort must be one of low, medium, high, max, xhigh")
		}
	}

	return nil
}
