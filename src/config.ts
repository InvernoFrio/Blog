import type {
	ExpressiveCodeConfig,
	LicenseConfig,
	NavBarConfig,
	ProfileConfig,
	SiteConfig,
} from "./types/config";
import { LinkPreset } from "./types/config";

export const siteConfig: SiteConfig = {
	title: "InvernoFrio的博客",
	subtitle: "一名热爱技术的开发者",
	lang: "zh_CN",
	themeColor: {
		hue: 200, // 蓝色调
		fixed: false,
	},
	banner: {
		enable: true,
		src: "assets/images/cover.jpg",
		position: "center 20%",
		credit: {
			enable: false,
			text: "",
			url: "",
		},
	},
	toc: {
		enable: true,
		depth: 2,
	},
	favicon: [
		{
			src: "/favicon.svg",
			theme: "light",
			sizes: "any",
		},
	],
};

export const navBarConfig: NavBarConfig = {
	links: [
		LinkPreset.Home,
		LinkPreset.Archive,
		LinkPreset.About,
		{
			name: "友链",
			url: "/friend-links/",
		},
		{
			name: "游戏",
			url: "/games/",
		},
		{
			name: "小满",
			url: "/chat/",
		},
		{
			name: "GitHub",
			url: "https://github.com/invernofrio",
			external: true,
		},
	],
};

export const profileConfig: ProfileConfig = {
	avatar: "assets/images/avatar.jpg",
	name: "InvernoFrio",
	bio: "一名热爱技术的开发者，喜欢折腾服务器、写代码、偶尔写点东西记录生活",
	links: [
		{
			name: "GitHub",
			icon: "fa6-brands:github",
			url: "https://github.com/invernofrio",
		},
		{
			name: "Email",
			icon: "fa6-solid:envelope",
			url: "mailto:1580757598@qq.com",
		},
	],
};

export const licenseConfig: LicenseConfig = {
	enable: true,
	name: "CC BY-NC-SA 4.0",
	url: "https://creativecommons.org/licenses/by-nc-sa/4.0/",
};

export const expressiveCodeConfig: ExpressiveCodeConfig = {
	theme: "github-dark",
};
