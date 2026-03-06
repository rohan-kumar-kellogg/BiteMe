'use client'

import { ImageIcon, Clock, Utensils } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import type { RecentUpload } from '@/lib/types'

interface RecentUploadsProps {
  uploads: RecentUpload[]
  onRemoveUpload?: (uploadId: string) => Promise<void>
}

export function RecentUploads({ uploads, onRemoveUpload }: RecentUploadsProps) {
  if (uploads.length === 0) {
    return (
      <section className="rounded-3xl bg-card border border-border/50 p-6 md:p-8 space-y-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
            <ImageIcon className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h3 className="text-lg font-semibold">Food History</h3>
            <p className="text-sm text-muted-foreground">Your culinary journey</p>
          </div>
        </div>
        
        <div className="text-center py-8">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-primary/10 flex items-center justify-center">
            <Utensils className="w-8 h-8 text-primary/50" />
          </div>
          <p className="text-muted-foreground mb-2">No uploads yet</p>
          <p className="text-sm text-muted-foreground/70">
            Start uploading food photos to build your taste profile
          </p>
        </div>
      </section>
    )
  }

  return (
    <section className="rounded-3xl bg-card border border-border/50 p-6 md:p-8 space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
            <ImageIcon className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h3 className="text-lg font-semibold">Food History</h3>
            <p className="text-sm text-muted-foreground">{uploads.length} dishes analyzed</p>
          </div>
        </div>
        <div className="flex items-center gap-1 text-xs text-muted-foreground">
          <Clock className="w-3.5 h-3.5" />
          <span>Recent</span>
        </div>
      </div>

      {/* Grid of uploads */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        {uploads.map((upload) => (
          <UploadCard key={upload.id} upload={upload} onRemoveUpload={onRemoveUpload} />
        ))}
      </div>
    </section>
  )
}

function UploadCard({
  upload,
  onRemoveUpload,
}: {
  upload: RecentUpload
  onRemoveUpload?: (uploadId: string) => Promise<void>
}) {
  return (
    <div className="group relative rounded-2xl overflow-hidden bg-muted aspect-square">
      {/* Placeholder image with gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-primary/20 to-primary/5" />
      <div className="absolute inset-0 flex items-center justify-center">
        <Utensils className="w-8 h-8 text-primary/30" />
      </div>
      
      {/* Overlay with info */}
      <div className="absolute inset-0 bg-gradient-to-t from-foreground/80 via-foreground/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity">
        <div className="absolute bottom-0 left-0 right-0 p-3">
          <h4 className="text-sm font-medium text-background truncate">
            {upload.dish_label}
          </h4>
          {upload.cuisine && (
            <p className="text-xs text-background/70">{upload.cuisine}</p>
          )}
        </div>
      </div>

      {/* Always visible label */}
      <div className="absolute bottom-0 left-0 right-0 p-3 bg-gradient-to-t from-foreground/60 to-transparent group-hover:opacity-0 transition-opacity">
        <h4 className="text-xs font-medium text-background truncate">
          {upload.dish_label}
        </h4>
      </div>

      {/* Tags */}
      {upload.tags && upload.tags.length > 0 && (
        <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
          <Badge variant="secondary" className="text-[10px] px-1.5 py-0.5">
            {upload.tags[0]}
          </Badge>
        </div>
      )}
      {onRemoveUpload && (
        <div className="absolute top-2 left-2 opacity-0 group-hover:opacity-100 transition-opacity">
          <Button
            variant="secondary"
            size="sm"
            className="h-6 px-2 text-[10px]"
            onClick={() => onRemoveUpload(upload.id)}
          >
            Remove
          </Button>
        </div>
      )}
    </div>
  )
}
