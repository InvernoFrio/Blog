import assert from "node:assert/strict";
import { readdir, readFile, stat } from "node:fs/promises";
import path from "node:path";

const topics = ["ai-infra/operators", "llm/from-scratch"];
const origin = "https://local.test";
const home = await readFile("dist/index.html", "utf8");
function assertTemplateLayout(html) {
	for (const id of [
		"top-row",
		"navbar-wrapper",
		"main-grid",
		"content-wrapper",
		"sidebar",
		"back-to-top-btn",
	]) {
		assert.equal(
			[...html.matchAll(new RegExp(`id="${id}"`, "g"))].length,
			1,
			`Learning pages must include the shared template's ${id}`,
		);
	}
	assert.equal(
		html.includes('id="banner-wrapper"'),
		home.includes('id="banner-wrapper"'),
		"Learning pages must follow the template banner setting",
	);
}
let links = 0;
let articleCount = 0;
for (const topic of topics) {
	const source = `src/content/learning/${topic}`;
	const lessons = await Promise.all(
		(await readdir(source))
			.filter((file) => file.endsWith(".md"))
			.map(async (file) => {
				const content = await readFile(path.join(source, file), "utf8");
				return {
					slug: file.slice(0, -3),
					order: Number(content.match(/^order: (\d+)$/m)[1]),
					headings: [...content.matchAll(/^#{2,3}\s+.+$/gm)].length,
				};
			}),
	);
	lessons.sort((a, b) => a.order - b.order);
	assert.equal(
		new Set(lessons.map((lesson) => lesson.order)).size,
		lessons.length,
		"Lesson order must be unique within a topic",
	);
	articleCount += lessons.length;
	for (const [index, lesson] of lessons.entries()) {
		const route = `/Blog/learn/${topic}/${lesson.slug}/`;
		const html = await readFile(
			`dist/learn/${topic}/${lesson.slug}/index.html`,
			"utf8",
		);
		assertTemplateLayout(html);
		assert.match(
			html,
			/<body[^>]*data-no-swup/,
			"Learning pages need a complete layout initialization",
		);
		assert.equal([...html.matchAll(/id="swup-container"/g)].length, 1);
		assert.equal([...html.matchAll(/id="toc"/g)].length, 1);
		assert.match(html, /<h1 data-pagefind-meta="title"/);
		const article = html.match(
			/<article[^>]*data-pagefind-body[^>]*>([\s\S]*?)<\/article>/,
		)[1];
		assert.equal(
			[...article.matchAll(/<h[23]\b/g)].length,
			lesson.headings,
			`Rendered article must retain every source section: ${route}`,
		);
		assert(
			!html.includes('class="katex-error"'),
			`Math must render without KaTeX errors: ${route}`,
		);
		const navigation = html.match(
			new RegExp(
				`<nav aria-label="[^"]+" data-learning-topic="${topic}">([\\s\\S]*?)<\\/nav>`,
			),
		)[1];
		const articleLinks = [
			...navigation.matchAll(/href="([^"]+)"(?: aria-current="page")?/g),
		].slice(1);
		assert.equal(articleLinks.length, lessons.length);
		for (const match of articleLinks)
			assert(
				match[1].startsWith(`/Blog/learn/${topic}/`),
				"Left navigation must stay within the active topic",
			);
		assert.equal([...navigation.matchAll(/aria-current="page"/g)].length, 1);
		assert(
			navigation.includes(`href="${route}" aria-current="page"`),
			"The current article must be marked in the left navigation",
		);
		const ids = new Set(
			[...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]),
		);
		const outline = html.match(
			/<nav aria-label="本页章节目录">([\s\S]*?)<\/nav>/,
		)[1];
		for (const match of outline.matchAll(/href="#([^"]+)"/g)) {
			assert(
				ids.has(decodeURIComponent(match[1])),
				`Outline target is missing: ${match[1]}`,
			);
		}
		if (home.includes('id="toc-wrapper"')) {
			const desktopOutline = html.match(
				/<table-of-contents\b[^>]*>([\s\S]*?)<\/table-of-contents>/,
			)?.[1];
			assert(
				desktopOutline,
				"Learning articles must use the template desktop outline",
			);
			for (const match of desktopOutline.matchAll(/href="#([^"]+)"/g)) {
				assert(
					ids.has(decodeURIComponent(match[1])),
					`Template outline target is missing: ${match[1]}`,
				);
			}
		}
		for (const [relation, target] of [
			["prev", lessons[index - 1]],
			["next", lessons[index + 1]],
		]) {
			if (target)
				assert(
					html.includes(
						`href="/Blog/learn/${topic}/${target.slug}/" rel="${relation}"`,
					),
				);
			else assert(!html.includes(`rel="${relation}"`));
		}
		for (const match of html.matchAll(/(?:href|src)="([^"<>]+)"/g)) {
			const value = match[1].replaceAll("&amp;", "&");
			if (value.startsWith("#") || value.startsWith("data:")) continue;
			const target = new URL(value, origin + route);
			if (target.origin !== origin) continue;
			assert(
				target.pathname.startsWith("/Blog/"),
				`Link loses the deployment base: ${value}`,
			);
			let local = path.join(
				"dist",
				decodeURIComponent(target.pathname.slice(6)),
			);
			if (target.pathname.endsWith("/")) local = path.join(local, "index.html");
			await assert.doesNotReject(
				stat(local),
				`Missing resource: ${target.pathname}`,
			);
			links++;
		}
	}
}
assert.match(home, /aria-label="学习" href="\/Blog\/learn\/"/);
for (const route of ["learn", "learn/ai-infra", "learn/llm"]) {
	const html = await readFile(`dist/${route}/index.html`, "utf8");
	assertTemplateLayout(html);
	assert.match(html, /data-no-swup/);
}
assert.match(
	await readFile("dist/learn/llm/from-scratch/probability/index.html", "utf8"),
	/class="katex/,
);
for (const file of await readdir("public/learning/llm-from-scratch")) {
	assert.deepEqual(
		await readFile(`dist/learning/llm-from-scratch/${file}`),
		await readFile(`public/learning/llm-from-scratch/${file}`),
		`Published experiment asset must match its source: ${file}`,
	);
}
assert(
	!(await readFile("dist/rss.xml", "utf8")).includes("/learn/"),
	"Lessons must stay separate from the chronological blog feed",
);
assert.match(
	await readFile("dist/learn/ai-infra/operators/softmax/index.html", "utf8"),
	/class="katex/,
);
for (const file of ["cpu_checks.py", "kernels.py", "gpu_lab.py"]) {
	assert.equal(
		await readFile(`dist/learning/operators/${file}`, "utf8"),
		await readFile(`public/learning/operators/${file}`, "utf8"),
	);
}
console.log(
	`Learning build checks passed: ${articleCount} articles across ${topics.length} topics, isolated navigation, current-page markers, outline anchors, previous/next links, ${links} local resources, math, downloads, and separate RSS.`,
);
