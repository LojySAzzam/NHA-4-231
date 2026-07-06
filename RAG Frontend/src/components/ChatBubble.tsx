import { Bot, ThumbsDown, ThumbsUp, User } from "lucide-react";
import { formatTime } from "../lib/utils";
import type { ChatMessage } from "../types/chat";
import { SourceCard } from "./SourceCard";
import { TypingIndicator } from "./TypingIndicator";

interface ChatBubbleProps {
  message: ChatMessage;
  onFeedback: (messageId: string, value: "up" | "down") => void;
  onTypingDone: (messageId: string) => void;
}

export function ChatBubble({ message, onFeedback, onTypingDone }: ChatBubbleProps) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <article className="ml-auto flex max-w-[92%] items-end gap-1.5 sm:max-w-[80%] sm:gap-2">
        <div className="min-w-0 flex-1 rounded-2xl rounded-br-sm bg-primary px-3 py-2.5 text-sm leading-relaxed text-primary-foreground shadow-sm sm:px-4 sm:py-3">
          <p className="break-words whitespace-pre-wrap [overflow-wrap:anywhere]">{message.content}</p>
          <p className="mt-1 text-right text-[11px] text-primary-foreground/75">
            {formatTime(message.createdAt)}
          </p>
        </div>
        <span className="hidden h-8 w-8 shrink-0 items-center justify-center rounded-full border border-border bg-card text-muted-foreground sm:inline-flex">
          <User className="h-4 w-4" />
        </span>
      </article>
    );
  }

  return (
    <article className="flex w-full max-w-full items-start gap-1.5 sm:max-w-[90%] sm:gap-2">
      <span className="mt-1 inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground sm:h-8 sm:w-8">
        <Bot className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
      </span>

      <div className="min-w-0 flex-1 space-y-2">
        <div className="rounded-2xl rounded-tl-sm border border-border bg-card px-3 py-2.5 shadow-sm sm:px-4 sm:py-3">
          {message.isTyping ? (
            <TypingIndicator
              text={message.content}
              onComplete={() => onTypingDone(message.id)}
            />
          ) : (
            <p className="break-words whitespace-pre-wrap text-sm leading-relaxed text-foreground [overflow-wrap:anywhere]">
              {message.content}
            </p>
          )}

          {message.sources && message.sources.length > 0 ? (
            <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2">
              {message.sources.map((source) => (
                <SourceCard
                  key={`${source.title}-${source.page}`}
                  source={source}
                />
              ))}
            </div>
          ) : null}
        </div>

        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={() => onFeedback(message.id, "up")}
            className={`inline-flex h-8 w-8 items-center justify-center rounded-lg border transition ${
              message.feedback === "up"
                ? "border-emerald-300 bg-emerald-100 text-emerald-700 dark:border-emerald-800 dark:bg-emerald-950/50 dark:text-emerald-300"
                : "border-border bg-card text-muted-foreground hover:text-foreground"
            }`}
            aria-label="Helpful response"
            title="Helpful"
          >
            <ThumbsUp className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={() => onFeedback(message.id, "down")}
            className={`inline-flex h-8 w-8 items-center justify-center rounded-lg border transition ${
              message.feedback === "down"
                ? "border-red-300 bg-red-100 text-red-700 dark:border-red-800 dark:bg-red-950/50 dark:text-red-300"
                : "border-border bg-card text-muted-foreground hover:text-foreground"
            }`}
            aria-label="Not helpful response"
            title="Not helpful"
          >
            <ThumbsDown className="h-4 w-4" />
          </button>
          <span className="ml-2 text-[11px] text-muted-foreground">
            {formatTime(message.createdAt)}
          </span>
        </div>
      </div>
    </article>
  );
}
