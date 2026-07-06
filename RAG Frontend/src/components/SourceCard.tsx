import { FileText } from "lucide-react";
import type { Source } from "../types/chat";

interface SourceCardProps {
  source: Source;
}

export function SourceCard({ source }: SourceCardProps) {
  return (
    <article className="rounded-xl border border-border bg-muted/30 p-3">
      <div className="flex items-center gap-2">
        <span className="inline-flex h-8 w-8 items-center justify-center rounded-lg bg-accent text-accent-foreground">
          <FileText className="h-4 w-4" />
        </span>
        <div className="min-w-0">
          <p className="truncate text-sm font-medium text-foreground">{source.title}</p>
          <p className="text-xs text-muted-foreground">Page {source.page}</p>
        </div>
      </div>
    </article>
  );
}
