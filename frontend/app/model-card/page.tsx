import { promises as fs } from 'fs'
import path from 'path'
import Link from 'next/link'
import { ArrowLeft } from 'lucide-react'
import { ModelCardTabs, type ModelCardSection } from '@/components/model-card-tabs'
import { Button } from '@/components/ui/button'
import { BiteMeLogo } from '@/components/bite-me-logo'

const SECTION_TITLES = [
  'Model Overview',
  'Intended Use',
  'Data',
  'Evaluation',
  'Performance & Limitations',
  'Improvement Path',
  'Summary',
] as const

function sectionKey(title: string): string {
  return title.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')
}

function parseModelCard(raw: string): { title: string; subtitle: string; sections: ModelCardSection[] } {
  const lines = raw.split(/\r?\n/)
  const title = (lines[0] || 'BiteMe Model Card').trim()
  const subtitle = (lines[1] || 'Taste Inference & Food Understanding System').trim()
  const sectionsMap = new Map<string, string[]>()
  SECTION_TITLES.forEach((s) => sectionsMap.set(s, []))

  let current: string | null = null
  for (const lineRaw of lines.slice(2)) {
    const line = String(lineRaw || '')
    const trimmed = line.trim()
    const found = SECTION_TITLES.find((s) => s === trimmed)
    if (found) {
      current = found
      continue
    }
    if (!current) continue
    sectionsMap.get(current)?.push(line)
  }

  const sections: ModelCardSection[] = SECTION_TITLES.map((title) => ({
    key: sectionKey(title),
    title,
    lines: sectionsMap.get(title) || [],
  }))
  return { title, subtitle, sections }
}

export default async function ModelCardPage() {
  const modelCardPath = path.resolve(process.cwd(), '..', 'docs', 'model_card.md')
  const raw = await fs.readFile(modelCardPath, 'utf-8')
  const parsed = parseModelCard(raw)

  return (
    <div className="min-h-screen bg-background">
      <header className="sticky top-0 z-50 bg-background/80 backdrop-blur-lg border-b border-border/50">
        <div className="max-w-6xl mx-auto px-4 h-14 flex items-center justify-between">
          <Button asChild variant="ghost" size="sm" className="gap-2 -ml-2">
            <Link href="/">
              <ArrowLeft className="w-4 h-4" />
              Back
            </Link>
          </Button>
          <BiteMeLogo iconSize={28} />
          <div className="w-16" />
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8 md:py-12 space-y-8">
        <section className="rounded-3xl border border-border/60 bg-card/80 backdrop-blur-sm shadow-xl shadow-black/[0.03] p-6 md:p-10">
          <p className="text-xs uppercase tracking-[0.22em] text-primary/80 font-medium mb-3">Model Documentation</p>
          <h1 className="text-3xl md:text-4xl font-semibold tracking-tight text-balance">{parsed.title}</h1>
          <p className="text-base md:text-lg text-muted-foreground mt-2">{parsed.subtitle}</p>
          <p className="text-sm text-muted-foreground mt-5 max-w-3xl leading-7">
            This page documents what the BiteMe AI system does, how it works, where it performs well, and where it has
            limitations. Content is loaded directly from the existing model card source.
          </p>
        </section>

        <ModelCardTabs sections={parsed.sections} defaultKey={sectionKey('Model Overview')} />
      </main>
    </div>
  )
}

