import { type CollectionEntry, getCollection } from "astro:content";
import { url } from "./url-utils";

export type Lesson = CollectionEntry<"learning">;
export interface LearningTopic {
	course: string;
	courseTitle: string;
	topic: string;
	title: string;
}
export const infraOverviewTopic: LearningTopic = {
	course: "ai-infra",
	courseTitle: "AI Infra",
	topic: "foundations",
	title: "总体概览：AI Infra 研究什么",
};
export const operatorTopic: LearningTopic = {
	course: "ai-infra",
	courseTitle: "AI Infra",
	topic: "operators",
	title: "算子：从原理到实现",
};
export const inferenceTopic: LearningTopic = {
	course: "ai-infra",
	courseTitle: "AI Infra",
	topic: "inference",
	title: "LLM 推理：从算子到服务",
};
export const llmTrainingTopic: LearningTopic = {
	course: "llm",
	courseTitle: "LLM",
	topic: "from-scratch",
	title: "从零训练 LLM",
};
export const learningTopics = [
	infraOverviewTopic,
	operatorTopic,
	inferenceTopic,
	llmTrainingTopic,
];
export function getLessonTopic(lesson: Lesson) {
	const topic = learningTopics.find(
		(item) =>
			item.course === lesson.data.course && item.topic === lesson.data.topic,
	);
	if (!topic)
		throw new Error(
			`Unknown learning topic: ${lesson.data.course}/${lesson.data.topic}`,
		);
	return topic;
}
export async function getTopicLessons(topic: LearningTopic) {
	return (
		await getCollection(
			"learning",
			({ data }) => data.course === topic.course && data.topic === topic.topic,
		)
	).sort((a, b) => a.data.order - b.data.order);
}
export async function getOperatorLessons() {
	return getTopicLessons(operatorTopic);
}
export function lessonUrl(lesson: Lesson) {
	return url(`/learn/${lesson.slug}/`);
}
