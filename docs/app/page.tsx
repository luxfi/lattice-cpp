import Link from 'next/link';

export default function HomePage() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-8 text-center">
      <div className="max-w-3xl">
        <h1 className="mb-4 text-5xl font-bold tracking-tight">lux-lattice</h1>
        <p className="mb-8 text-xl text-gray-600 dark:text-gray-400">
          GPU-accelerated lattice cryptography library for post-quantum cryptography.
          Built on lux-gpu for optimal performance across Metal, CUDA, and CPU backends.
        </p>
        <div className="flex flex-wrap justify-center gap-4">
          <Link
            href="/docs"
            className="rounded-lg bg-blue-600 px-6 py-3 font-medium text-white transition hover:bg-blue-700"
          >
            Get Started
          </Link>
          <a
            href="https://github.com/luxfi/lattice"
            className="rounded-lg border border-gray-300 px-6 py-3 font-medium transition hover:bg-gray-100 dark:border-gray-700 dark:hover:bg-gray-800"
          >
            GitHub
          </a>
        </div>
        <div className="mt-16 grid gap-8 md:grid-cols-3">
          <div className="rounded-lg border border-gray-200 p-6 dark:border-gray-800">
            <h3 className="mb-2 text-lg font-semibold">🔢 NTT Operations</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              GPU-accelerated Number Theoretic Transform for fast polynomial multiplication
            </p>
          </div>
          <div className="rounded-lg border border-gray-200 p-6 dark:border-gray-800">
            <h3 className="mb-2 text-lg font-semibold">🔐 Ring-LWE</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Polynomial ring operations in R_q = Z_q[X]/(X^n + 1) for threshold signatures
            </p>
          </div>
          <div className="rounded-lg border border-gray-200 p-6 dark:border-gray-800">
            <h3 className="mb-2 text-lg font-semibold">⚡ GPU Backends</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Metal (Apple), CUDA (NVIDIA), and optimized CPU fallback via lux-gpu
            </p>
          </div>
        </div>
      </div>
    </main>
  );
}
