import Link from "next/link";

export default function NotFound() {
  return (
    <main
      id="main-content"
      className="mx-auto flex min-h-[70vh] max-w-3xl flex-col justify-center px-5 py-16 sm:px-8"
    >
      <p className="editorial-kicker">404 / Not found</p>
      <h1 className="page-title mt-5">这一页不在目录里。</h1>
      <p className="page-description mt-6">
        论文可能已移除，或地址已经变化。回到论文库继续查找。
      </p>
      <Link
        href="/library"
        className="mt-8 inline-flex min-h-12 w-fit items-center bg-[var(--foreground)] px-6 text-sm font-semibold text-white"
      >
        返回论文库
      </Link>
    </main>
  );
}
