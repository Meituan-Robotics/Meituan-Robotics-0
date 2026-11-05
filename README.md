# Meituan-Robotics-0 网页交接包

整理日期：2026-11-03。基于当前已发布源码 c6b1440。

本包可独立建立新 Git 仓库，不含原仓库 `.git`、登录信息、依赖目录和本机缓存。原网站不会因此变动。

## 内容

- `app/`：页面、样式、视频播放器及放大预览。
- `public/assets/`：SVG 与视频封面。
- `public/video/`：网页使用的 6 个 MP4，保持现有清晰度，并非全部拍摄原片。
- `public/Meituan-Robotics-0.pdf`：当前网页提供的论文。
- `data/project-results.json`：当前指标，含 RoboDojo SR 14.07%，小米位于 MR0 左侧。
- `.github/workflows/deploy-pages.yml`：自动构建和发布 GitHub Pages。

Vision-Language Data 最新图为 `vision-language-data-20261030.svg`，Overview 为 `main-overview-20261029.svg`。旧 SVG 保留用于归档检查。`data/archived-overview-results.json` 仅用于旧图，当前网页指标请修改 `data/project-results.json`。

## 本地运行

建议 macOS/Linux，或 Windows 的 WSL；现有命令使用 POSIX 环境变量语法。安装 Node.js 24 与 pnpm 11.19.0。在本文件所在目录执行：

```bash
pnpm install --frozen-lockfile
pnpm run dev -- --port 3000
```

浏览器打开 http://localhost:3000/ 。首次运行需要联网下载依赖。当前展示功能不需要 API Key、数据库或原作者账户。

## 发布到自己的 GitHub

1. 创建自己的空 GitHub 仓库（如 `mr0-project`），不要预先创建 README。
2. 在本目录执行，替换远程地址：

```bash
git init
git add .
git commit -m "Initial project page"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

3. 仓库 Settings → Pages → Build and deployment → Source 选择 **GitHub Actions**。
4. 在 Actions 中查看 `Deploy GitHub Pages`。首次推送若因 Pages 未开启失败，完成第 3 步后重新运行。
5. 成功后打开 `https://你的用户名.github.io/你的仓库名/`，以部署结果链接为准。

配置兼容 `main` 和 `master`，自动读取当前仓库名和账户，无需原作者账户。适用于普通项目仓库；`用户名.github.io` 根站点仓库、自定义域名需另外调整路径。私有仓库发布 Pages 的可用性取决于账户方案和组织权限。

请通过 Git 上传，不建议用网页逐个拖入：视频较大，且 `.github`、`.openai` 等隐藏目录容易漏传。当前单个视频均小于 100 MiB，不需要额外设置 Git LFS。

## 构建检查

```bash
pnpm run build
node --test tests/rendered-html.test.mjs tests/project-results.test.mjs tests/figure-viewer.test.mjs
```

Actions 会进一步导出静态网页并检查资源。不要直接双击 `app/page.tsx`；它不是可直接打开的 HTML。

手动导出时，以仓库名 `mr0-project` 为例，先执行：

```bash
GITHUB_PAGES_BASE_PATH=/mr0-project pnpm run build
GITHUB_PAGES_BASE_PATH=/mr0-project pnpm run start -- --port 3001 --hostname 127.0.0.1
```

保持服务运行，在另一个终端执行（替换用户名）：

```bash
GITHUB_PAGES_BASE_PATH=/mr0-project GITHUB_REPOSITORY=YOUR-USERNAME/mr0-project GITHUB_REPOSITORY_OWNER=YOUR-USERNAME pnpm run export:github-pages
GITHUB_PAGES_BASE_PATH=/mr0-project GITHUB_REPOSITORY=YOUR-USERNAME/mr0-project GITHUB_REPOSITORY_OWNER=YOUR-USERNAME node --test tests/github-pages-export.test.mjs
```

静态输出位于 `.github-pages/`，不随源码提交，会在发布时重新生成。

## 注意

不包含 PPT 原稿、视频原片备份、旧中英对照草稿或 `node_modules`。依赖由锁文件安装。

`.openai/hosting.json` 为构建兼容配置，原 Site ID 已清空；GitHub Pages 不使用原 Sites 服务。

当前站点保留原来的 `noindex` 设置，直接访问不受影响。如需被搜索引擎收录，另行检查 `app/layout.tsx`、`scripts/export-github-pages.mjs` 和 `public/robots.txt`。
