import { useEffect, useMemo, useState } from "react";

interface TypingIndicatorProps {
  text: string;
  speed?: number;
  onComplete?: () => void;
}

export function TypingIndicator({
  text,
  speed = 14,
  onComplete,
}: TypingIndicatorProps) {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    setIndex(0);
  }, [text]);

  useEffect(() => {
    if (index >= text.length) {
      onComplete?.();
      return;
    }

    const timer = window.setTimeout(() => {
      setIndex((current) => current + 1);
    }, speed);

    return () => window.clearTimeout(timer);
  }, [index, speed, text.length, onComplete]);

  const visibleText = useMemo(() => text.slice(0, index), [text, index]);

  return (
    <p className="whitespace-pre-wrap text-sm leading-relaxed text-foreground">
      {visibleText}
      {index < text.length ? <span className="animate-pulse">|</span> : null}
    </p>
  );
}
