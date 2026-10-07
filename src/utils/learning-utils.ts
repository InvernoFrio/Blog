import { type CollectionEntry, getCollection } from "astro:content";
import { url } from "./url-utils";

export type Lesson = CollectionEntry<"learning">;
export async function getOperatorLessons() {
	return (
		await getCollection(
			"learning",
			({ data }) => data.course === "ai-infra" && data.topic === "operators",
		)
	).sort((a, b) => a.data.order - b.data.order);
}
export function lessonUrl(lesson: Lesson) {
	return url(`/learn/${lesson.slug}/`);
}
