import { useEffect, useMemo, useRef } from "react";
import type { ChatMessage } from "../types/chat";
import { ChatBubble } from "./ChatBubble";
import { LoadingBubble } from "./LoadingBubble";
import { SuggestedQuestions } from "./SuggestedQuestions";

interface ChatWindowProps {
  messages: ChatMessage[];
  onFeedback: (messageId: string, value: "up" | "down") => void;
  onTypingDone: (messageId: string) => void;
  onSuggest?: (question: string) => void;
}

export function ChatWindow({ messages, onFeedback, onTypingDone, onSuggest }: ChatWindowProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const hasContent = messages.length > 0;
  const hasLoadingBubble = useMemo(
    () => messages.some((message) => message.isLoading),
    [messages]
  );

  if (!hasContent) {
    return (
      <div className="flex h-full min-h-0 flex-col items-center justify-center px-3 py-6 sm:px-4 sm:py-8">
        <div className="w-full max-w-xl rounded-2xl border border-border bg-card p-5 text-center shadow-sm sm:rounded-3xl sm:p-8">
          <p className="text-base font-semibold text-foreground sm:text-lg">Start the conversation</p>
          <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
            Ask a question and the assistant will answer using your REST API response and show related
            sources.
          </p>
        </div>
        {onSuggest ? (
          <div className="mt-6 w-full max-w-xl min-w-0">
            <SuggestedQuestions onSelect={onSuggest} />
          </div>
        ) : null}
      </div>
    );
  }

  return (
    <div className="mx-auto w-full min-w-0 max-w-3xl space-y-3 px-3 py-3 sm:space-y-4 sm:px-4 sm:py-4 lg:px-6">
      {messages
        .filter((message) => !message.isLoading)
        .map((message) => (
          <ChatBubble
            key={message.id}
            message={message}
            onFeedback={onFeedback}
            onTypingDone={onTypingDone}
          />
        ))}

      {hasLoadingBubble ? <LoadingBubble /> : null}

      <div ref={bottomRef} />
    </div>
  );
}
