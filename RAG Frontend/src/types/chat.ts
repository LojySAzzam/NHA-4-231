export type ChatRole = "user" | "assistant";

export type FeedbackValue = "up" | "down" | null;

export interface Source {
  title: string;
  page: number;
}

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  createdAt: number;
  sources?: Source[];
  feedback?: FeedbackValue;
  isLoading?: boolean;
  isTyping?: boolean;
}

export interface ChatSession {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: ChatMessage[];
}

export interface ChatApiRequest {
  message: string;
}

export interface ChatApiResponse {
  answer: string;
  sources: Source[];
}

export interface ChatStoragePayload {
  sessions: ChatSession[];
  activeSessionId: string;
}
