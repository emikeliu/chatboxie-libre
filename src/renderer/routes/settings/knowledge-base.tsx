import { createFileRoute } from '@tanstack/react-router'
import KnowledgeBase from '@/components/knowledge-base/KnowledgeBase'

export const Route = createFileRoute('/settings/knowledge-base')({ component: RouteComponent })

export function RouteComponent() {
  return <KnowledgeBase />
}
