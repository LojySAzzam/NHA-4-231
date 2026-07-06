import { Bot } from "lucide-react";

export function LoadingBubble() {
  return (
    <div className="flex items-start gap-2 sm:gap-3">
      <div className="mt-1 inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
        <Bot className="h-4 w-4" />
      </div>
      <div className="rounded-2xl rounded-tl-sm border border-border bg-card px-4 py-3 shadow-sm">
        <div className="flex items-center gap-1.5">
          <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground/80 [animation-delay:0ms]" />
          <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground/80 [animation-delay:140ms]" />
          <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground/80 [animation-delay:280ms]" />
        </div>
      </div>
    </div>
  );
}
