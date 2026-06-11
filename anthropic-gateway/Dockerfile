FROM golang:1.22-alpine AS build

WORKDIR /src

COPY go.mod ./
COPY main.go ./

RUN go build -o /out/anthropic-gateway .

FROM alpine:3.20

RUN apk add --no-cache ca-certificates wget

WORKDIR /app

COPY --from=build /out/anthropic-gateway /app/anthropic-gateway

EXPOSE 8081

CMD ["/app/anthropic-gateway"]
