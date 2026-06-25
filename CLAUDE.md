# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a personal blog built on the [Fuwari](https://github.com/saicaca/fuwari) Astro template. It's deployed to GitHub Pages at `https://invernofrio.github.io/Blog/`.

## Tech Stack

- **Framework**: Astro 5.x with Svelte components
- **Styling**: Tailwind CSS + Stylus (`.styl` files)
- **Package Manager**: pnpm (enforced via `preinstall` script)
- **Linting/Formatting**: Biome
- **Search**: Pagefind (runs after build)
- **Markdown**: remark/rehype plugins for math (KaTeX), admonitions, GitHub cards, reading time

## Commands

```bash
pnpm dev              # Start dev server at localhost:4321
pnpm build            # Build for production (includes pagefind indexing)
pnpm preview          # Preview production build locally
pnpm check            # Run Astro type checking
pnpm format           # Format code with Biome
pnpm lint             # Lint and auto-fix with Biome
pnpm new-post <name>  # Create new post in src/content/posts/
```

## Architecture

### Configuration
- `src/config.ts` — Main blog config (site title, nav links, profile, theme)
- `src/types/config.ts` — TypeScript types for all config objects
- `astro.config.mjs` — Astro build config, integrations, and Markdown plugins

### Content
- `src/content/posts/` — Blog posts as Markdown files with frontmatter
- `src/content/spec/` — Special pages (about, contact, friend-links, games)
- `src/content/config.ts` — Zod schemas for content collections

### Key Directories
- `src/components/` — UI components (Astro `.astro` + Svelte `.svelte`)
- `src/layouts/` — Page layouts (`MainGridLayout.astro` is the primary)
- `src/pages/` — File-based routing
- `src/plugins/` — Custom remark/rehype/Expressive Code plugins
- `src/styles/` — Global styles (CSS + Stylus)
- `src/i18n/` — Translations (site supports en, zh_CN, zh_TW, ja, ko, es, th, vi, tr, id)
- `src/utils/` — Utilities for URLs, dates, content processing, settings

### Deployment
GitHub Actions workflow (`.github/workflows/deploy.yml`) builds and deploys to GitHub Pages on push to `main`.

## Post Frontmatter

```yaml
---
title: Post Title
published: 2024-01-01
description: ''
image: ''
tags: [Tag1, Tag2]
category: ''
draft: false
lang: ''  # Only set if post language differs from site default
---
```

## Code Style

- Biome enforces: tab indentation, double quotes, recommended lint rules
- `.svelte` and `.astro` files have relaxed lint rules (unused vars/imports allowed)
- CSS files in `src/` are excluded from Biome formatting
- Use Conventional Commits format for commit messages

## Content Configuration

Edit `frontmatter.json` to configure the Front Matter CMS extension (VS Code).
