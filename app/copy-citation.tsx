"use client";

import { useEffect, useRef, useState } from "react";

type CopyCitationProps = {
  text: string;
};

type CopyStatus = "idle" | "copied" | "failed";

const labels: Record<CopyStatus, string> = {
  idle: "Copy Citation",
  copied: "Copied",
  failed: "Copy failed",
};

export default function CopyCitation({ text }: CopyCitationProps) {
  const [status, setStatus] = useState<CopyStatus>("idle");
  const resetTimer = useRef<number | null>(null);

  useEffect(
    () => () => {
      if (resetTimer.current !== null) window.clearTimeout(resetTimer.current);
    },
    [],
  );

  async function copyCitation() {
    if (resetTimer.current !== null) window.clearTimeout(resetTimer.current);

    try {
      if (!navigator.clipboard) throw new Error("Clipboard API unavailable");
      await navigator.clipboard.writeText(text);
      setStatus("copied");
    } catch {
      setStatus("failed");
    }

    resetTimer.current = window.setTimeout(() => {
      setStatus("idle");
      resetTimer.current = null;
    }, 2000);
  }

  return (
    <button
      type="button"
      className="copy-citation"
      onClick={copyCitation}
      data-status={status === "failed" ? "error" : status}
      aria-live="polite"
      aria-atomic="true"
    >
      {labels[status]}
    </button>
  );
}
