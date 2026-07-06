interface SuggestedQuestionsProps {
  onSelect: (question: string) => void;
  compact?: boolean;
}

const QUESTIONS = [
  "What is your return policy?",
  "How can I reset my password?",
  "Do you support enterprise plans?",
  "How do I track my order?",
  "What payment methods are accepted?",
  "Can you summarize shipping options?",
];

export function SuggestedQuestions({ onSelect, compact = false }: SuggestedQuestionsProps) {
  return (
    <section className="min-w-0 space-y-2 sm:space-y-3">
      <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground sm:text-xs">
        Suggested Questions
      </p>
      <div
        className={
          compact
            ? "flex gap-2 overflow-x-auto pb-1 [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
            : "flex flex-wrap gap-2"
        }
      >
        {QUESTIONS.map((question) => (
          <button
            key={question}
            type="button"
            onClick={() => onSelect(question)}
            className={`shrink-0 rounded-full border border-border bg-card text-foreground transition hover:border-primary/40 hover:bg-accent/40 ${
              compact
                ? "max-w-[calc(100vw-2.5rem)] truncate px-3 py-1.5 text-xs"
                : "px-3 py-1.5 text-sm sm:max-w-none"
            }`}
          >
            {question}
          </button>
        ))}
      </div>
    </section>
  );
}
