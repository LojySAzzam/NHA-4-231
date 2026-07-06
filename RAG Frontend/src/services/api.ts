import type { ChatApiRequest, ChatApiResponse } from "../types/chat";

const DEFAULT_CHAT_PATH = "/api/chat";

function getConfiguredEndpoint(): string {
  const baseUrl = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim();

  if (!baseUrl) {
    return "";
  }

  const normalizedBase = baseUrl.replace(/\/+$/, "");
  return `${normalizedBase}${DEFAULT_CHAT_PATH}`;
}

function buildEndpointCandidates(): string[] {
  const configured = getConfiguredEndpoint();
  if (!configured) {
    return [DEFAULT_CHAT_PATH];
  }

  // In development, try local proxy path as a fallback when direct host fetch fails.
  if (configured !== DEFAULT_CHAT_PATH) {
    return [configured, DEFAULT_CHAT_PATH];
  }

  return [configured];
}

async function readErrorDetails(response: Response): Promise<string> {
  const contentType = response.headers.get("content-type") || "";

  if (contentType.includes("application/json")) {
    const json = (await response.json().catch(() => null)) as
      | { message?: unknown; error?: unknown; detail?: unknown }
      | null;

    if (json) {
      const parts = [json.message, json.error, json.detail]
        .filter((value): value is string => typeof value === "string" && value.trim().length > 0)
        .map((value) => value.trim());

      if (parts.length > 0) {
        return parts.join(" | ");
      }
    }
  }

  return response.text().catch(() => "");
}

function getRequestTraceId(response: Response): string {
  const candidates = [
    response.headers.get("x-request-id"),
    response.headers.get("x-correlation-id"),
    response.headers.get("trace-id"),
    response.headers.get("x-amzn-trace-id"),
  ];

  const match = candidates.find(
    (value): value is string => typeof value === "string" && value.trim().length > 0
  );

  return match?.trim() ?? "";
}

async function sendToEndpoint(
  endpoint: string,
  payload: ChatApiRequest,
  signal?: AbortSignal
): Promise<ChatApiResponse> {
  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok) {
    const details = await readErrorDetails(response);
    const traceId = getRequestTraceId(response);
    const statusInfo = `${response.status}${response.statusText ? ` ${response.statusText}` : ""}`;
    const traceSuffix = traceId ? ` | request-id: ${traceId}` : "";

    if (response.status === 404) {
      throw new Error(
        `API endpoint not found (404): ${endpoint}. Configure VITE_API_BASE_URL to your backend host.`
      );
    }

    if (response.status === 500) {
      throw new Error(
        details
          ? `Backend internal error (${statusInfo}) at ${endpoint}: ${details}${traceSuffix}`
          : `Backend internal error (${statusInfo}) at ${endpoint}. The server returned an empty error body. Check backend logs using timestamp ${new Date().toISOString()}${traceSuffix}`
      );
    }

    throw new Error(
      details
        ? `Request failed (${statusInfo}) at ${endpoint}: ${details}${traceSuffix}`
        : `Request failed (${statusInfo}) at ${endpoint}.${traceSuffix}`
    );
  }

  const data = (await response.json()) as Partial<ChatApiResponse>;

  if (typeof data.answer !== "string") {
    throw new Error("Invalid API response: missing answer");
  }

  const sources = Array.isArray(data.sources)
    ? data.sources
        .filter(
          (item): item is { title: string; page: number } =>
            typeof item?.title === "string" && typeof item?.page === "number"
        )
        .map((source) => ({
          title: source.title,
          page: source.page,
        }))
    : [];

  return {
    answer: data.answer,
    sources,
  };
}

export async function sendChatMessage(
  message: string,
  signal?: AbortSignal
): Promise<ChatApiResponse> {
  const payload: ChatApiRequest = { message };
  const endpoints = buildEndpointCandidates();
  let lastError: unknown = null;

  for (const endpoint of endpoints) {
    try {
      return await sendToEndpoint(endpoint, payload, signal);
    } catch (error) {
      lastError = error;
      if (error instanceof Error && !error.message.toLowerCase().includes("failed to fetch")) {
        throw error;
      }
    }
  }

  const configured = getConfiguredEndpoint();
  if (lastError instanceof TypeError || lastError instanceof Error) {
    const attempted = endpoints.join(" , ");
    throw new Error(
      `Unable to reach backend API. Attempted: ${attempted}. Check that your backend is running, VITE_API_BASE_URL is correct, and CORS allows this origin.`
    );
  }

  if (!configured) {
    throw new Error("Backend API is not reachable. Set VITE_API_BASE_URL in .env.local.");
  }

  throw new Error("Unable to send chat request.");
}
