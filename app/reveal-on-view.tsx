"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";

type RevealOnViewProps = Readonly<{
  children: ReactNode;
  className?: string;
}>;

export default function RevealOnView({
  children,
  className = "",
}: RevealOnViewProps) {
  const rootRef = useRef<HTMLDivElement>(null);
  const [isRevealed, setIsRevealed] = useState(false);
  const [isAnimated, setIsAnimated] = useState(false);

  useEffect(() => {
    const element = rootRef.current;
    if (!element) return;

    let isMounted = true;

    const prefersReducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;

    if (prefersReducedMotion || !("IntersectionObserver" in window)) {
      queueMicrotask(() => {
        if (isMounted) setIsRevealed(true);
      });
      return () => {
        isMounted = false;
      };
    }

    queueMicrotask(() => {
      if (isMounted) setIsAnimated(true);
    });

    const observer = new IntersectionObserver(
      (entries) => {
        if (!entries.some((entry) => entry.isIntersecting)) return;

        if (isMounted) setIsRevealed(true);
        observer.disconnect();
      },
      { threshold: 0.2 },
    );

    observer.observe(element);
    return () => {
      isMounted = false;
      observer.disconnect();
    };
  }, []);

  const classes = [
    className,
    isAnimated ? "is-animated" : "",
    isRevealed ? "is-revealed" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div ref={rootRef} className={classes || undefined}>
      {children}
    </div>
  );
}
