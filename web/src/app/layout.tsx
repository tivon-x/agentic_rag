import type { Metadata } from "next";
import Link from "next/link";

import { text } from "@/lib/i18n";

import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "Paper Index",
    template: "%s | Paper Index",
  },
  description: "可搜索、可校正、可回到 PDF 原页的个人论文库。",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="zh-CN"
      className="h-full antialiased"
      data-scroll-behavior="smooth"
    >
      <body className="min-h-full">
        <a href="#main-content" className="skip-link">
          跳到正文
        </a>
        <div className="app-shell">
          <header className="border-b border-[var(--line)] bg-[var(--panel)]">
            <div className="mx-auto flex w-full max-w-7xl items-center justify-between gap-4 px-5 py-4 sm:px-8">
              <Link
                href="/"
                className="font-serif text-lg font-semibold tracking-tight text-slate-950"
              >
                {text.nav.brand}
              </Link>
              <nav aria-label="主导航" className="flex items-center gap-1 sm:gap-4">
                <Link href="/library" className="nav-link">
                  {text.nav.library}
                </Link>
                <Link href="/search" className="nav-link">
                  {text.nav.search}
                </Link>
                <Link href="/chat" className="nav-link">
                  {text.nav.chat}
                </Link>
              </nav>
            </div>
          </header>
          {children}
        </div>
      </body>
    </html>
  );
}
