'use client'

import { useMemo } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'

export type ModelCardSection = {
  key: string
  title: string
  lines: string[]
}

type Block =
  | { type: 'heading'; text: string }
  | { type: 'paragraph'; text: string }
  | { type: 'list-item'; text: string }

function isHeadingLike(line: string): boolean {
  if (!line || line.endsWith('.') || line.endsWith(':')) return false
  if (line.length > 60) return false
  const words = line.split(/\s+/)
  if (words.length > 6) return false
  return /^[A-Z0-9][A-Za-z0-9&/\- ]+$/.test(line)
}

function isListItemLike(line: string): boolean {
  if (!line || line.endsWith('.') || line.endsWith(':')) return false
  if (line.length > 80) return false
  if (isHeadingLike(line)) return false
  return true
}

function toBlocks(lines: string[]): Block[] {
  const blocks: Block[] = []
  let chunk: string[] = []
  const flush = () => {
    if (chunk.length === 0) return
    const text = chunk.join(' ').trim()
    if (!text) {
      chunk = []
      return
    }
    if (chunk.length === 1 && isHeadingLike(text)) {
      blocks.push({ type: 'heading', text })
    } else if (chunk.length === 1 && isListItemLike(text)) {
      blocks.push({ type: 'list-item', text })
    } else {
      blocks.push({ type: 'paragraph', text })
    }
    chunk = []
  }
  for (const raw of lines) {
    const line = String(raw || '').trim()
    if (!line) {
      flush()
      continue
    }
    chunk.push(line)
  }
  flush()
  return blocks
}

export function ModelCardTabs({
  sections,
  defaultKey,
}: {
  sections: ModelCardSection[]
  defaultKey: string
}) {
  const blocksBySection = useMemo(
    () => Object.fromEntries(sections.map((s) => [s.key, toBlocks(s.lines)])),
    [sections]
  )

  return (
    <Tabs defaultValue={defaultKey} className="gap-6">
      <div className="rounded-2xl border border-border/60 bg-card/80 backdrop-blur-sm shadow-sm p-2">
        <TabsList className="w-full h-auto bg-transparent p-0 flex overflow-x-auto justify-start">
          {sections.map((section) => (
            <TabsTrigger
              key={section.key}
              value={section.key}
              className="rounded-xl px-4 py-2.5 text-sm data-[state=active]:shadow-md transition-all duration-200"
            >
              {section.title}
            </TabsTrigger>
          ))}
        </TabsList>
      </div>

      {sections.map((section) => (
        <TabsContent
          key={section.key}
          value={section.key}
          className="mt-0 data-[state=active]:animate-in data-[state=active]:fade-in-0 data-[state=active]:slide-in-from-bottom-1"
        >
          <section className="rounded-3xl border border-border/60 bg-card/70 shadow-lg shadow-black/[0.03] p-6 md:p-8 transition-all duration-200">
            <h2 className="text-xl font-semibold tracking-tight mb-6">{section.title}</h2>
            <div className="space-y-4">
              {blocksBySection[section.key]?.map((block, idx) => {
                if (block.type === 'heading') {
                  return (
                    <h3 key={`${section.key}-h-${idx}`} className="text-sm font-semibold text-foreground/90 pt-1">
                      {block.text}
                    </h3>
                  )
                }
                if (block.type === 'list-item') {
                  return (
                    <div key={`${section.key}-li-${idx}`} className="flex items-start gap-2.5 text-sm text-muted-foreground">
                      <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-primary/70 shrink-0" />
                      <span>{block.text}</span>
                    </div>
                  )
                }
                return (
                  <p key={`${section.key}-p-${idx}`} className="text-sm leading-7 text-muted-foreground">
                    {block.text}
                  </p>
                )
              })}
            </div>
          </section>
        </TabsContent>
      ))}
    </Tabs>
  )
}

