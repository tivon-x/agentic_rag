import type { Metadata } from "next";

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
        {children}
      </body>
    </html>
  );
}
