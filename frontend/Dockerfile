# Stage 1: Build
FROM node:20-slim AS builder
WORKDIR /app
COPY package*.json .
RUN npm ci
COPY . .
ENV NEXT_TELEMETRY_DISABLED 1
RUN npm run build

# Stage 2: Runtime
FROM node:20-slim AS runtime
RUN groupadd -r neuroflow && useradd -r -g neuroflow neuroflow
WORKDIR /app
ENV NODE_ENV production
ENV NEXT_TELEMETRY_DISABLED 1
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public
RUN chown -R neuroflow:neuroflow /app
USER neuroflow
EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD node -e "require('http').get('http://localhost:3000', r => process.exit(r.statusCode < 500 ? 0 : 1))"
CMD ["node", "server.js"]
