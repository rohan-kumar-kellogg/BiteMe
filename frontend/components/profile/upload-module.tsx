'use client'

import { useState, useCallback } from 'react'
import { Upload, ImageIcon, X, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

interface UploadModuleProps {
  onUpload: (file: File) => Promise<void>
  isUploading: boolean
}

type UploadState = 'idle' | 'preview' | 'uploading' | 'success' | 'error'

export function UploadModule({ onUpload, isUploading }: UploadModuleProps) {
  const [state, setState] = useState<UploadState>('idle')
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [dragActive, setDragActive] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string>('')

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }, [])

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith('image/')) {
      setErrorMessage('Please upload an image file')
      setState('error')
      return
    }

    if (file.size > 10 * 1024 * 1024) {
      setErrorMessage('Image must be less than 10MB')
      setState('error')
      return
    }

    setSelectedFile(file)
    setPreviewUrl(URL.createObjectURL(file))
    setState('preview')
    setErrorMessage('')
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }, [handleFile])

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  const handleUpload = async () => {
    if (!selectedFile) return

    setState('uploading')
    try {
      await onUpload(selectedFile)
      setState('success')
      setTimeout(() => {
        resetState()
      }, 2000)
    } catch (error) {
      setErrorMessage('Upload failed. Please try again.')
      setState('error')
    }
  }

  const resetState = () => {
    setState('idle')
    setPreviewUrl(null)
    setSelectedFile(null)
    setErrorMessage('')
  }

  return (
    <section className="rounded-3xl bg-card border border-border/50 p-6 md:p-8 space-y-4">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
          <Upload className="w-5 h-5 text-primary" />
        </div>
        <div>
          <h3 className="text-lg font-semibold">Add to Your Profile</h3>
          <p className="text-sm text-muted-foreground">Upload a food photo to refine your taste</p>
        </div>
      </div>

      {/* Upload Area */}
      <div
        className={cn(
          'relative rounded-2xl border-2 border-dashed transition-all duration-200',
          state === 'idle' && 'border-border/50 hover:border-primary/50 hover:bg-primary/5',
          state === 'preview' && 'border-primary/50 bg-primary/5',
          state === 'uploading' && 'border-primary bg-primary/10',
          state === 'success' && 'border-chart-3 bg-chart-3/10',
          state === 'error' && 'border-destructive bg-destructive/5',
          dragActive && 'border-primary bg-primary/10 scale-[1.02]'
        )}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        {/* Idle State */}
        {state === 'idle' && (
          <label className="flex flex-col items-center justify-center py-12 px-6 cursor-pointer">
            <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center mb-4">
              <ImageIcon className="w-8 h-8 text-primary" />
            </div>
            <p className="text-sm font-medium mb-1">Drop your food photo here</p>
            <p className="text-xs text-muted-foreground mb-4">or click to browse</p>
            <Button variant="outline" size="sm" className="rounded-full" asChild>
              <span>Choose File</span>
            </Button>
            <input
              type="file"
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              accept="image/*"
              onChange={handleChange}
            />
          </label>
        )}

        {/* Preview State */}
        {state === 'preview' && previewUrl && (
          <div className="p-4">
            <div className="relative aspect-video rounded-xl overflow-hidden bg-muted mb-4">
              <img
                src={previewUrl}
                alt="Preview"
                className="w-full h-full object-cover"
              />
              <button
                onClick={resetState}
                className="absolute top-2 right-2 w-8 h-8 rounded-full bg-foreground/80 text-background flex items-center justify-center hover:bg-foreground transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="flex gap-2">
              <Button onClick={handleUpload} className="flex-1 rounded-full">
                Analyze This Dish
              </Button>
              <Button variant="outline" onClick={resetState} className="rounded-full">
                Cancel
              </Button>
            </div>
          </div>
        )}

        {/* Uploading State */}
        {state === 'uploading' && (
          <div className="flex flex-col items-center justify-center py-12 px-6">
            <div className="relative mb-4">
              <div className="w-16 h-16 rounded-full bg-primary/20 flex items-center justify-center">
                <Loader2 className="w-8 h-8 text-primary animate-spin" />
              </div>
            </div>
            <p className="text-sm font-medium mb-1">Analyzing your dish...</p>
            <p className="text-xs text-muted-foreground">This usually takes a few seconds</p>
          </div>
        )}

        {/* Success State */}
        {state === 'success' && (
          <div className="flex flex-col items-center justify-center py-12 px-6">
            <div className="w-16 h-16 rounded-full bg-chart-3/20 flex items-center justify-center mb-4">
              <CheckCircle2 className="w-8 h-8 text-chart-3" />
            </div>
            <p className="text-sm font-medium mb-1">Upload successful!</p>
            <p className="text-xs text-muted-foreground">Your profile has been updated</p>
          </div>
        )}

        {/* Error State */}
        {state === 'error' && (
          <div className="flex flex-col items-center justify-center py-12 px-6">
            <div className="w-16 h-16 rounded-full bg-destructive/10 flex items-center justify-center mb-4">
              <AlertCircle className="w-8 h-8 text-destructive" />
            </div>
            <p className="text-sm font-medium mb-1">Something went wrong</p>
            <p className="text-xs text-muted-foreground mb-4">{errorMessage}</p>
            <Button variant="outline" size="sm" onClick={resetState} className="rounded-full">
              Try Again
            </Button>
          </div>
        )}
      </div>
    </section>
  )
}
