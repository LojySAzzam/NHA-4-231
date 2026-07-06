import { SendHorizontal } from "lucide-react";
import { useEffect, useRef } from "react";

interface ChatInputProps {
  value: string;
  isSending: boolean;
  onChange: (value: string) => void;
  onSend: () => void;
}

export function ChatInput({ value, isSending, onChange, onSend }: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const element = textareaRef.current;
    if (!element) {
      return;
    }

    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, 180)}px`;
  }, [value]);

  return (
    <div className="shrink-0 border-t border-border bg-background/95 px-3 py-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] sm:px-4 lg:px-5">
      <div className="mx-auto flex w-full min-w-0 max-w-3xl items-end gap-2 rounded-2xl border border-border bg-card px-3 py-2.5 shadow-sm focus-within:border-primary/50 focus-within:ring-2 focus-within:ring-primary/20">
        <textarea
          ref={textareaRef}
          rows={1}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              onSend();
            }
          }}
          placeholder="Type your message..."
          disabled={isSending}
          className="max-h-[180px] min-h-[24px] min-w-0 flex-1 resize-none bg-transparent text-sm text-foreground outline-none placeholder:text-muted-foreground disabled:opacity-60"
        />

        <button
          type="button"
          disabled={isSending || !value.trim()}
          onClick={onSend}
          className="inline-flex h-10 w-10 items-center justify-center rounded-xl bg-primary text-primary-foreground transition hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50"
          aria-label="Send message"
          title="Send"
        >
          <SendHorizontal className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}
