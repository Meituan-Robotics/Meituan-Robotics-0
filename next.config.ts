import type { NextConfig } from "next";

const githubPagesBasePath = process.env.GITHUB_PAGES_BASE_PATH ?? "";

const nextConfig: NextConfig = {
  basePath: githubPagesBasePath || undefined,
  assetPrefix: githubPagesBasePath || undefined,
};

export default nextConfig;
