"use client"

import { useState, useRef, useCallback } from "react"
import { Camera, Upload, Trash2, Recycle, Leaf, AlertTriangle, RotateCcw } from 'lucide-react'
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"

interface ClassificationResult {
  category: 'organic' | 'recyclable' | 'hazardous'
  confidence: number
  subcategory: string
  disposalInstructions: string
}

export default function SmartWasteBin() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [isClassifying, setIsClassifying] = useState(false)
  const [classificationResult, setClassificationResult] = useState<ClassificationResult | null>(null)
  const [isCameraActive, setIsCameraActive] = useState(false)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Simulated CNN classification function
  const classifyWaste = useCallback(async (imageData: string): Promise<ClassificationResult> => {
    // Simulate processing time
    await new Promise(resolve => setTimeout(resolve, 2000))
    
    // Simulate CNN analysis based on image characteristics
    const classifications = [
      {
        category: 'organic' as const,
        confidence: Math.random() * 30 + 70,
        subcategory: 'Food Waste',
        disposalInstructions: 'Dispose in organic waste bin. Can be composted.'
      },
      {
        category: 'recyclable' as const,
        confidence: Math.random() * 25 + 75,
        subcategory: 'Plastic Bottle',
        disposalInstructions: 'Clean and place in recycling bin. Remove cap if different material.'
      },
      {
        category: 'recyclable' as const,
        confidence: Math.random() * 20 + 80,
        subcategory: 'Paper/Cardboard',
        disposalInstructions: 'Ensure clean and dry. Place in paper recycling bin.'
      },
      {
        category: 'hazardous' as const,
        confidence: Math.random() * 15 + 85,
        subcategory: 'Electronic Waste',
        disposalInstructions: 'Take to designated e-waste collection center. Do not dispose in regular bins.'
      },
      {
        category: 'hazardous' as const,
        confidence: Math.random() * 20 + 80,
        subcategory: 'Battery',
        disposalInstructions: 'Take to battery recycling center. Contains toxic materials.'
      }
    ]
    
    return classifications[Math.floor(Math.random() * classifications.length)]
  }, [])

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        const imageData = e.target?.result as string
        setSelectedImage(imageData)
        setClassificationResult(null)
      }
      reader.readAsDataURL(file)
    }
  }

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'environment' } 
      })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        setIsCameraActive(true)
      }
    } catch (error) {
      console.error('Error accessing camera:', error)
    }
  }

  const captureImage = () => {
    if (videoRef.current && canvasRef.current) {
      const canvas = canvasRef.current
      const video = videoRef.current
      const context = canvas.getContext('2d')
      
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      
      if (context) {
        context.drawImage(video, 0, 0)
        const imageData = canvas.toDataURL('image/jpeg')
        setSelectedImage(imageData)
        setClassificationResult(null)
        
        // Stop camera
        const stream = video.srcObject as MediaStream
        stream?.getTracks().forEach(track => track.stop())
        setIsCameraActive(false)
      }
    }
  }

  const handleClassify = async () => {
    if (!selectedImage) return
    
    setIsClassifying(true)
    try {
      const result = await classifyWaste(selectedImage)
      setClassificationResult(result)
    } catch (error) {
      console.error('Classification error:', error)
    } finally {
      setIsClassifying(false)
    }
  }

  const reset = () => {
    setSelectedImage(null)
    setClassificationResult(null)
    setIsCameraActive(false)
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream
      stream.getTracks().forEach(track => track.stop())
    }
  }

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'organic':
        return <Leaf className="w-6 h-6" />
      case 'recyclable':
        return <Recycle className="w-6 h-6" />
      case 'hazardous':
        return <AlertTriangle className="w-6 h-6" />
      default:
        return <Trash2 className="w-6 h-6" />
    }
  }

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'organic':
        return 'bg-green-500'
      case 'recyclable':
        return 'bg-blue-500'
      case 'hazardous':
        return 'bg-red-500'
      default:
        return 'bg-gray-500'
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50 p-4">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Smart Waste Classification Bin
          </h1>
          <p className="text-lg text-gray-600">
            AI-powered waste sorting using computer vision
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {/* Image Input Section */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Camera className="w-5 h-5" />
                Image Input
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {!selectedImage && !isCameraActive && (
                <div className="space-y-4">
                  <Button 
                    onClick={startCamera} 
                    className="w-full"
                    variant="outline"
                  >
                    <Camera className="w-4 h-4 mr-2" />
                    Use Camera
                  </Button>
                  
                  <div className="relative">
                    <Button 
                      onClick={() => fileInputRef.current?.click()}
                      className="w-full"
                      variant="outline"
                    >
                      <Upload className="w-4 h-4 mr-2" />
                      Upload Image
                    </Button>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/*"
                      onChange={handleImageUpload}
                      className="hidden"
                    />
                  </div>
                </div>
              )}

              {isCameraActive && (
                <div className="space-y-4">
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    className="w-full rounded-lg"
                  />
                  <div className="flex gap-2">
                    <Button onClick={captureImage} className="flex-1">
                      Capture
                    </Button>
                    <Button onClick={reset} variant="outline">
                      Cancel
                    </Button>
                  </div>
                </div>
              )}

              {selectedImage && (
                <div className="space-y-4">
                  <img
                    src={selectedImage || "/placeholder.svg"}
                    alt="Selected waste item"
                    className="w-full rounded-lg object-cover max-h-64"
                  />
                  <div className="flex gap-2">
                    <Button 
                      onClick={handleClassify}
                      disabled={isClassifying}
                      className="flex-1"
                    >
                      {isClassifying ? 'Analyzing...' : 'Classify Waste'}
                    </Button>
                    <Button onClick={reset} variant="outline">
                      <RotateCcw className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              )}

              <canvas ref={canvasRef} className="hidden" />
            </CardContent>
          </Card>

          {/* Classification Results Section */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Trash2 className="w-5 h-5" />
                Classification Results
              </CardTitle>
            </CardHeader>
            <CardContent>
              {isClassifying && (
                <div className="space-y-4">
                  <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
                    <p className="text-gray-600">Analyzing waste item...</p>
                  </div>
                  <Progress value={33} className="w-full" />
                </div>
              )}

              {classificationResult && (
                <div className="space-y-6">
                  <div className="text-center">
                    <div className={`inline-flex items-center justify-center w-16 h-16 rounded-full ${getCategoryColor(classificationResult.category)} text-white mb-4`}>
                      {getCategoryIcon(classificationResult.category)}
                    </div>
                    <h3 className="text-2xl font-bold capitalize mb-2">
                      {classificationResult.category}
                    </h3>
                    <Badge variant="secondary" className="text-sm">
                      {classificationResult.subcategory}
                    </Badge>
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">Confidence</span>
                      <span className="text-sm text-gray-600">
                        {classificationResult.confidence.toFixed(1)}%
                      </span>
                    </div>
                    <Progress value={classificationResult.confidence} className="w-full" />
                  </div>

                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h4 className="font-semibold mb-2">Disposal Instructions:</h4>
                    <p className="text-sm text-gray-700">
                      {classificationResult.disposalInstructions}
                    </p>
                  </div>
                </div>
              )}

              {!isClassifying && !classificationResult && (
                <div className="text-center text-gray-500 py-8">
                  <Trash2 className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>Upload or capture an image to classify waste</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Information Cards */}
        <div className="grid md:grid-cols-3 gap-4 mt-8">
          <Card className="border-green-200">
            <CardContent className="p-4 text-center">
              <div className="bg-green-500 text-white rounded-full w-12 h-12 flex items-center justify-center mx-auto mb-3">
                <Leaf className="w-6 h-6" />
              </div>
              <h3 className="font-semibold text-green-800">Organic Waste</h3>
              <p className="text-sm text-green-600 mt-1">
                Food scraps, yard waste, biodegradable materials
              </p>
            </CardContent>
          </Card>

          <Card className="border-blue-200">
            <CardContent className="p-4 text-center">
              <div className="bg-blue-500 text-white rounded-full w-12 h-12 flex items-center justify-center mx-auto mb-3">
                <Recycle className="w-6 h-6" />
              </div>
              <h3 className="font-semibold text-blue-800">Recyclable</h3>
              <p className="text-sm text-blue-600 mt-1">
                Plastic, paper, glass, metal containers
              </p>
            </CardContent>
          </Card>

          <Card className="border-red-200">
            <CardContent className="p-4 text-center">
              <div className="bg-red-500 text-white rounded-full w-12 h-12 flex items-center justify-center mx-auto mb-3">
                <AlertTriangle className="w-6 h-6" />
              </div>
              <h3 className="font-semibold text-red-800">Hazardous</h3>
              <p className="text-sm text-red-600 mt-1">
                Batteries, electronics, chemicals, medical waste
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
