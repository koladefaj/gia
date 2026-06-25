'use client';

/**
 * TypingText — types out a sequence of phrases with a blinking caret, deleting
 * and cycling to the next. Used for the landing tagline ("Your voice. Your
 * music." → other value props).
 */

import { useEffect, useRef, useState } from 'react';

interface Props {
  phrases: string[];
  typeMs?: number;
  deleteMs?: number;
  holdMs?: number;
  className?: string;
}

export default function TypingText({
  phrases,
  typeMs = 65,
  deleteMs = 32,
  holdMs = 1600,
  className,
}: Props) {
  const [text, setText] = useState('');
  const idx = useRef(0);
  const deleting = useRef(false);

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;

    const tick = () => {
      const full = phrases[idx.current % phrases.length];
      const len = text.length;

      if (!deleting.current) {
        if (len < full.length) {
          setText(full.slice(0, len + 1));
          timer = setTimeout(tick, typeMs);
        } else {
          deleting.current = true;
          timer = setTimeout(tick, holdMs);
        }
      } else if (len > 0) {
        setText(full.slice(0, len - 1));
        timer = setTimeout(tick, deleteMs);
      } else {
        deleting.current = false;
        idx.current += 1;
        timer = setTimeout(tick, typeMs);
      }
    };

    timer = setTimeout(tick, typeMs);
    return () => clearTimeout(timer);
    // Re-run the scheduler whenever `text` changes (drives each keystroke).
  }, [text, phrases, typeMs, deleteMs, holdMs]);

  return (
    <span className={className}>
      {text}
      <span className="gia-caret" aria-hidden>
        |
      </span>
    </span>
  );
}
