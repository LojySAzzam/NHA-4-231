import { useCallback, useEffect, useMemo, useState } from "react";
import { makeId } from "../lib/utils";
import { sendChatMessage } from "../services/api";
import type { ChatMessage, ChatSession, ChatStoragePayload, FeedbackValue } from "../types/chat";

const STORAGE_KEY = "rag-frontend.chat.storage.v1";
const FALLBACK_ERROR = "I could not fetch an answer right now. Please try again.";

function createEmptySession(): ChatSession {
  const now = Date.now();
  return {
    id: makeId(),
    title: "New Chat",
    createdAt: now,
    updatedAt: now,
    messages: [],
  };
}

function parseStorage(raw: string | null): ChatStoragePayload | null {
  if (!raw) {
    return null;
  }

  try {
    const parsed = JSON.parse(raw) as Partial<ChatStoragePayload>;
    if (!Array.isArray(parsed.sessions) || typeof parsed.activeSessionId !== "string") {
      return null;
    }

    const sessions = parsed.sessions.filter(
      (session): session is ChatSession =>
        typeof session?.id === "string" &&
        typeof session?.title === "string" &&
        typeof session?.createdAt === "number" &&
        typeof session?.updatedAt === "number" &&
        Array.isArray(session?.messages)
    );

    if (sessions.length === 0) {
      return null;
    }

    return {
      sessions,
      activeSessionId: parsed.activeSessionId,
    };
  } catch {
    return null;
  }
}

export function useChat() {
  const [sessions, setSessions] = useState<ChatSession[]>(() => {
    const parsed = parseStorage(localStorage.getItem(STORAGE_KEY));
    return parsed?.sessions ?? [createEmptySession()];
  });

  const [activeSessionId, setActiveSessionId] = useState<string>(() => {
    const parsed = parseStorage(localStorage.getItem(STORAGE_KEY));
    return parsed?.activeSessionId ?? "";
  });

  const [isRequesting, setIsRequesting] = useState(false);

  useEffect(() => {
    if (!activeSessionId && sessions.length > 0) {
      setActiveSessionId(sessions[0].id);
    }
  }, [activeSessionId, sessions]);

  useEffect(() => {
    if (!activeSessionId) {
      return;
    }

    const payload: ChatStoragePayload = {
      sessions,
      activeSessionId,
    };

    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  }, [sessions, activeSessionId]);

  const activeSession = useMemo(
    () => sessions.find((session) => session.id === activeSessionId) ?? sessions[0],
    [sessions, activeSessionId]
  );

  const createNewSession = useCallback(() => {
    const newSession = createEmptySession();
    setSessions((prev) => [newSession, ...prev]);
    setActiveSessionId(newSession.id);
  }, []);

  const clearActiveSession = useCallback(() => {
    if (!activeSession) {
      return;
    }

    setSessions((prev) =>
      prev.map((session) =>
        session.id === activeSession.id
          ? {
              ...session,
              title: "New Chat",
              updatedAt: Date.now(),
              messages: [],
            }
          : session
      )
    );
  }, [activeSession]);

  const deleteSession = useCallback(
    (id: string) => {
      setSessions((prev) => {
        const next = prev.filter((session) => session.id !== id);
        if (next.length === 0) {
          const empty = createEmptySession();
          setActiveSessionId(empty.id);
          return [empty];
        }

        if (id === activeSessionId) {
          setActiveSessionId(next[0].id);
        }

        return next;
      });
    },
    [activeSessionId]
  );

  const setFeedback = useCallback(
    (messageId: string, value: Exclude<FeedbackValue, null>) => {
      if (!activeSession) {
        return;
      }

      setSessions((prev) =>
        prev.map((session) => {
          if (session.id !== activeSession.id) {
            return session;
          }

          const nextMessages = session.messages.map((message) => {
            if (message.id !== messageId) {
              return message;
            }

            const nextValue: FeedbackValue = message.feedback === value ? null : value;
            return {
              ...message,
              feedback: nextValue,
            };
          });

          return {
            ...session,
            messages: nextMessages,
            updatedAt: Date.now(),
          };
        })
      );
    },
    [activeSession]
  );

  const markTypingComplete = useCallback(
    (messageId: string) => {
      if (!activeSession) {
        return;
      }

      setSessions((prev) =>
        prev.map((session) => {
          if (session.id !== activeSession.id) {
            return session;
          }

          return {
            ...session,
            messages: session.messages.map((message) =>
              message.id === messageId
                ? {
                    ...message,
                    isTyping: false,
                  }
                : message
            ),
          };
        })
      );
    },
    [activeSession]
  );

  const sendMessage = useCallback(
    async (text: string) => {
      const content = text.trim();
      if (!content || isRequesting || !activeSession) {
        return;
      }

      const now = Date.now();
      const userMessage: ChatMessage = {
        id: makeId(),
        role: "user",
        content,
        createdAt: now,
      };

      const loadingMessage: ChatMessage = {
        id: makeId(),
        role: "assistant",
        content: "",
        createdAt: now + 1,
        isLoading: true,
      };

      setSessions((prev) =>
        prev.map((session) => {
          if (session.id !== activeSession.id) {
            return session;
          }

          return {
            ...session,
            title: session.messages.length === 0 ? content.slice(0, 40) : session.title,
            updatedAt: Date.now(),
            messages: [...session.messages, userMessage, loadingMessage],
          };
        })
      );

      setIsRequesting(true);

      try {
        const result = await sendChatMessage(content);

        setSessions((prev) =>
          prev.map((session) => {
            if (session.id !== activeSession.id) {
              return session;
            }

            const nextMessages = session.messages
              .filter((message) => message.id !== loadingMessage.id)
              .concat({
                id: makeId(),
                role: "assistant",
                content: result.answer,
                sources: result.sources,
                createdAt: Date.now(),
                feedback: null,
                isTyping: true,
              });

            return {
              ...session,
              messages: nextMessages,
              updatedAt: Date.now(),
            };
          })
        );
      } catch (error) {
        const message = error instanceof Error ? error.message : FALLBACK_ERROR;

        setSessions((prev) =>
          prev.map((session) => {
            if (session.id !== activeSession.id) {
              return session;
            }

            const nextMessages = session.messages
              .filter((entry) => entry.id !== loadingMessage.id)
              .concat({
                id: makeId(),
                role: "assistant",
                content: message || FALLBACK_ERROR,
                createdAt: Date.now(),
                feedback: null,
                isTyping: true,
              });

            return {
              ...session,
              messages: nextMessages,
              updatedAt: Date.now(),
            };
          })
        );
      } finally {
        setIsRequesting(false);
      }
    },
    [activeSession, isRequesting]
  );

  const sortedSessions = useMemo(
    () => [...sessions].sort((a, b) => b.updatedAt - a.updatedAt),
    [sessions]
  );

  return {
    sessions: sortedSessions,
    activeSession,
    activeSessionId,
    isRequesting,
    createNewSession,
    clearActiveSession,
    deleteSession,
    setActiveSessionId,
    sendMessage,
    setFeedback,
    markTypingComplete,
  };
}
