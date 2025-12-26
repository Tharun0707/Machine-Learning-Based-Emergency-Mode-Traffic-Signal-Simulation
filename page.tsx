"use client"

import type React from "react"
import { useState, useEffect, useRef } from "react"
import { Upload, Play, Pause, RotateCcw, AlertCircle, Truck } from "lucide-react"

type TrafficState = "red" | "yellow" | "green" | "ambulance-green" | "off"

interface DetectionFrame {
  timestamp: number
  confidence: number
}

interface ContinuousDetectionResult {
  file_type: string
  duration: number
  total_frames_processed: number
  emergency_detected: boolean
  total_detections: number
  max_confidence: number
  detection_frames: DetectionFrame[]
  message: string
}

interface DetectionResult {
  detected: boolean
  confidence: number
  detections: any[]
  message: string
}

// Simple Button component
function Button({
  children,
  onClick,
  disabled = false,
  variant = "default",
  className = "",
}: {
  children: React.ReactNode
  onClick?: () => void
  disabled?: boolean
  variant?: "default" | "outline"
  className?: string
}) {
  const baseClasses =
    "inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 h-10 px-4 py-2"
  const variantClasses =
    variant === "outline"
      ? "border border-gray-300 bg-transparent hover:bg-gray-50 text-gray-700"
      : "bg-blue-600 text-white hover:bg-blue-700"

  return (
    <button className={`${baseClasses} ${variantClasses} ${className}`} onClick={onClick} disabled={disabled}>
      {children}
    </button>
  )
}

// Simple Card components
function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <div className={`rounded-lg border bg-white shadow-sm ${className}`}>{children}</div>
}

function CardHeader({ children }: { children: React.ReactNode }) {
  return <div className="flex flex-col space-y-1.5 p-6">{children}</div>
}

function CardTitle({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <h3 className={`text-2xl font-semibold leading-none tracking-tight ${className}`}>{children}</h3>
}

function CardContent({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <div className={`p-6 pt-0 ${className}`}>{children}</div>
}

// Simple Alert components
function Alert({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`relative w-full rounded-lg border p-4 ${className}`} role="alert">
      {children}
    </div>
  )
}

function AlertDescription({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <div className={`text-sm ${className}`}>{children}</div>
}

// Traffic Light component
function TrafficLight({
  state,
  timeRemaining,
  ambulanceDetected,
}: {
  state: TrafficState
  timeRemaining: number
  ambulanceDetected: boolean
}) {
  const isRed = state === "red"
  const isYellow = state === "yellow"
  const isGreen = state === "green" || state === "ambulance-green"
  const isAmbulanceGreen = state === "ambulance-green"
  const isOff = state === "off"

  return (
    <div className="flex flex-col items-center space-y-6">
      {/* Traffic Light Housing */}
      <div className="bg-gray-800 rounded-3xl p-6 shadow-2xl border-4 border-gray-700">
        <div className="space-y-4">
          {/* Red Light */}
          <div className="relative">
            <div
              className={`w-20 h-20 rounded-full border-4 border-gray-600 transition-all duration-300 ${
                isRed ? "bg-red-500 shadow-red-glow border-red-400" : "bg-red-900/30"
              }`}
            />
            {isRed && <div className="absolute inset-0 w-20 h-20 rounded-full bg-red-400 animate-pulse opacity-50" />}
          </div>

          {/* Yellow Light */}
          <div className="relative">
            <div
              className={`w-20 h-20 rounded-full border-4 border-gray-600 transition-all duration-300 ${
                isYellow ? "bg-yellow-400 shadow-yellow-glow border-yellow-300" : "bg-yellow-900/30"
              }`}
            />
            {isYellow && (
              <div className="absolute inset-0 w-20 h-20 rounded-full bg-yellow-300 animate-pulse opacity-50" />
            )}
          </div>

          {/* Green Light */}
          <div className="relative">
            <div
              className={`w-20 h-20 rounded-full border-4 border-gray-600 transition-all duration-300 flex items-center justify-center ${
                isGreen ? "bg-green-500 shadow-green-glow border-green-400" : "bg-green-900/30"
              }`}
            />
            {isGreen && (
              <>
                <div className="absolute inset-0 w-20 h-20 rounded-full bg-green-400 animate-pulse opacity-50" />
                {isAmbulanceGreen && (
                  <div className="absolute inset-0 flex items-center justify-center z-10">
                    <Truck className="w-8 h-8 text-white animate-bounce" />
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>

      {/* Status Display */}
      <div className="bg-white rounded-lg p-4 shadow-lg border min-w-[200px]">
        <div className="text-center space-y-2">
          <div className="text-2xl font-mono font-bold text-gray-800">{isOff ? "--" : `${timeRemaining}s`}</div>
          <div
            className={`text-sm font-semibold uppercase tracking-wide ${
              isOff
                ? "text-gray-500"
                : state === "red"
                  ? "text-red-600"
                  : state === "yellow"
                    ? "text-yellow-600"
                    : state === "ambulance-green"
                      ? "text-orange-600"
                      : "text-green-600"
            }`}
          >
            {isOff ? "System Off" : state === "ambulance-green" ? "Emergency Mode" : state}
          </div>
          {state === "ambulance-green" && !isOff &&  (
            <div className="flex items-center justify-center gap-1 text-orange-600 text-xs">
              <Truck className="w-3 h-3" />
              <span>Emergency Priority</span>
            </div>
          )}
        </div>
      </div>

      {/* Traffic Light Pole */}
      <div className="w-4 h-32 bg-gray-600 rounded-full shadow-lg" />
      <div className="w-16 h-4 bg-gray-700 rounded-full shadow-lg" />
    </div>
  )
}

export default function EmergencyVehicleDetectionApp() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [detectionResult, setDetectionResult] = useState<ContinuousDetectionResult | null>(null)
  const [trafficState, setTrafficState] = useState<TrafficState>("off")
  const [timeRemaining, setTimeRemaining] = useState(0)
  const [isRunning, setIsRunning] = useState(false)
  const [systemStarted, setSystemStarted] = useState(false)
  const [apiError, setApiError] = useState<string | null>(null)
  const [processingProgress, setProcessingProgress] = useState("")
  const ambulanceDetected = detectionResult?.emergency_detected === true

  const timerRef = useRef<number | null>(null)
const ambulanceTimerRef = useRef<number | null>(null)

  // Normal traffic light cycle
  useEffect(() => {
    if (!isRunning || !systemStarted || ambulanceDetected) return

    timerRef.current = window.setInterval(() => {
      setTimeRemaining((prev) => {
        if (prev <= 1) {
          // Switch to next state and set appropriate duration
          setTrafficState((currentState) => {
            let newState: TrafficState
            let newDuration: number

            switch (currentState) {
              case "red":
                newState = "yellow"
                newDuration = 3 // Yellow for 3 seconds
                break
              case "yellow":
                newState = "green"
                newDuration = 30 // Green for 30 seconds
                break
              case "green":
                newState = "red"
                newDuration = 30 // Red for 30 seconds
                break
              default:
                newState = "red"
                newDuration = 30
            }

            // Set the duration for the NEW state
            window.setTimeout(() => setTimeRemaining(newDuration), 0)
            return newState
          })

          return 0 // This will be overridden by setTimeout above
        }
        return prev - 1
      })
    }, 1000)

    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [isRunning, systemStarted, ambulanceDetected])

  // Handle ambulance detection
  useEffect(() => {
    if (ambulanceDetected && systemStarted) {
      // Clear normal timer
      if (timerRef.current) clearInterval(timerRef.current)

      // Immediately turn green for ambulance
      setTrafficState("ambulance-green")
      setTimeRemaining(15)

      // Set 15-second timer for ambulance green
      ambulanceTimerRef.current = window.setTimeout(() => {
        setTrafficState("red")
        setTimeRemaining(30)
      }, 15000)
    }

    return () => {
      if (ambulanceTimerRef.current) clearTimeout(ambulanceTimerRef.current)
    }
  }, [ambulanceDetected, systemStarted])

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      const url = URL.createObjectURL(file)
      setPreviewUrl(url)
      setApiError(null)
      setDetectionResult(null)
      setProcessingProgress("")
    }
  }

  const detectEmergencyVehicle = async () => {
    if (!selectedFile) return

    setIsProcessing(true)
    setApiError(null)
    setProcessingProgress("Starting analysis...")

    try {
      // Create FormData to send the file
      const formData = new FormData()

// Determine if it's a video or image
const isVideo = selectedFile.type.startsWith("video/")

if (isVideo) {
  // Backend /detect-continuous expects "file"
  formData.append("file", selectedFile)
  setProcessingProgress("Processing video frames...")
} else {
  // Backend /detect expects "image"
  formData.append("image", selectedFile)
  setProcessingProgress("Analyzing image...")
}


      // Call the appropriate API endpoint
      const endpoint = isVideo ? "/detect-continuous" : "/detect"
      const response = await fetch(`http://localhost:5000${endpoint}`, {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`)
      }

      const result: ContinuousDetectionResult = await response.json()
      setDetectionResult(result)

      // Start the traffic system if not already started
      if (!systemStarted) {
        setSystemStarted(true)
        setIsRunning(true)
        setTrafficState("red")
        setTimeRemaining(30)
      }

      if (result.emergency_detected) {
        setSystemStarted(true)
        setIsRunning(true)
        setTrafficState("ambulance-green")
        setTimeRemaining(15)
}

      // Set emergency vehicle detection based on model result

      if (isVideo && result.emergency_detected) {
        setProcessingProgress(`Emergency vehicle detected in ${result.total_detections} frames!`)
      } else if (result.emergency_detected) {
        setProcessingProgress("Emergency vehicle detected!")
      } else {
        setProcessingProgress("No emergency vehicles detected")
      }
    } catch (error) {
      console.error("Detection error:", error)
      setApiError(error instanceof Error ? error.message : "Failed to connect to detection API")
      setProcessingProgress("")
    } finally {
      setIsProcessing(false)
    }
  }

  const resetSystem = () => {
    setTrafficState("off")
    setTimeRemaining(0)
    setIsRunning(false)
    setSystemStarted(false)
    setSelectedFile(null)
    setPreviewUrl(null)
    setIsProcessing(false)
    setDetectionResult(null)
    setApiError(null)
    setProcessingProgress("")
  }

  const toggleSystem = () => {
    if (systemStarted) {
      setIsRunning(!isRunning)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Smart Traffic Control System</h1>
          <p className="text-lg text-gray-600">AI-Powered Emergency Vehicle Detection & Traffic Light Control</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Input Section */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Upload className="w-5 h-5" />
                  Upload Media for Detection
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-gray-400 transition-colors">
                  <input
                    type="file"
                    accept="image/*,video/*"
                    onChange={handleFileSelect}
                    className="hidden"
                    id="file-upload"
                  />
                  <label htmlFor="file-upload" className="cursor-pointer">
                    <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                    <p className="text-lg font-medium text-gray-700 mb-2">Click to upload image or video</p>
                    <p className="text-sm text-gray-500">Supports JPG, PNG, MP4, MOV files</p>
                  </label>
                </div>

                {previewUrl && (
                  <div className="mt-4">
                    <div className="relative rounded-lg overflow-hidden bg-gray-100">
                      {selectedFile?.type.startsWith("image/") ? (
                        <img
                          src={previewUrl || "/placeholder.svg"}
                          alt="Preview"
                          className="w-full h-64 object-cover"
                        />
                      ) : (
                        <video src={previewUrl} controls className="w-full h-64 object-cover" />
                      )}
                    </div>
                    <p className="text-sm text-gray-600 mt-2">File: {selectedFile?.name}</p>
                  </div>
                )}

                {processingProgress && (
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                    <p className="text-blue-800 text-sm">{processingProgress}</p>
                  </div>
                )}

                {apiError && (
                  <Alert className="border-red-200 bg-red-50">
                    <div className="flex items-start gap-2">
                      <AlertCircle className="h-4 w-4 text-red-600 mt-0.5" />
                      <AlertDescription className="text-red-800">
                        {apiError}. Make sure the Python API server is running on localhost:5000
                      </AlertDescription>
                    </div>
                  </Alert>
                )}

                <div className="flex gap-3">
                  <Button onClick={detectEmergencyVehicle} disabled={!selectedFile || isProcessing} className="flex-1">
                    {isProcessing ? (
                      <>
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                        Analyzing...
                      </>
                    ) : (
                      "Emergency Vehicle Detection"
                    )}
                  </Button>
                  <Button onClick={resetSystem} variant="outline" className="flex items-center gap-2 bg-transparent">
                    <RotateCcw className="w-4 h-4" />
                    Reset
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Detection Results */}
            {detectionResult && (
              <Card>
                <CardHeader>
                  <CardTitle>Detection Results</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="font-medium">Result:</span>
                    <span
                      className={`font-bold ${detectionResult.emergency_detected ? "text-orange-600" : "text-green-600"}`}
                    >
                      {detectionResult.message}
                    </span>
                  </div>

                  {detectionResult.file_type === "video" && (
                    <>
                      <div className="flex items-center justify-between">
                        <span className="font-medium">Video Duration:</span>
                        <span className="font-mono">{detectionResult.duration.toFixed(1)}s</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="font-medium">Frames Processed:</span>
                        <span className="font-bold">{detectionResult.total_frames_processed}</span>
                      </div>
                    </>
                  )}

                  {detectionResult.emergency_detected && (
                    <>
                      <div className="flex items-center justify-between">
                        <span className="font-medium">Max Confidence:</span>
                        <span className="font-mono text-lg font-bold text-orange-600">
                          {(detectionResult.max_confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="font-medium">Detection Count:</span>
                        <span className="font-bold text-orange-600">{detectionResult.total_detections}</span>
                      </div>
                    </>
                  )}
                </CardContent>
              </Card>
            )}

            {/* Status Panel */}
            <Card>
              <CardHeader>
                <CardTitle>System Status</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="font-medium">Traffic System:</span>
                  <div className="flex items-center gap-2">
                    <div
                      className={`w-3 h-3 rounded-full ${
                        systemStarted ? (isRunning ? "bg-green-500" : "bg-yellow-500") : "bg-gray-400"
                      }`}
                    />
                    <span
                      className={systemStarted ? (isRunning ? "text-green-600" : "text-yellow-600") : "text-gray-600"}
                    >
                      {!systemStarted ? "Waiting for Input" : isRunning ? "Running" : "Paused"}
                    </span>
                  </div>
                </div>

                <div className="flex items-center justify-between">
  <span className="font-medium">Emergency Vehicle:</span>
  <div className="flex items-center gap-2">
    <div
      className={`w-3 h-3 rounded-full ${
        detectionResult?.emergency_detected ? "bg-green-500" : "bg-gray-300"
      }`}
    />
    <span
      className={
        detectionResult?.emergency_detected ? "text-green-600" : "text-gray-600"
      }
    >
      {detectionResult?.emergency_detected ? "DETECTED" : "NOT DETECTED"}
    </span>
  </div>
</div>



                <div className="flex items-center justify-between">
                  <span className="font-medium">Current State:</span>
                  <span className="capitalize font-semibold">
                    {trafficState === "off" ? "System Off" : trafficState.replace("-", " ")}
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="font-medium">Time Remaining:</span>
                  <span className="font-mono text-lg font-bold">
                    {trafficState === "off" ? "--" : `${timeRemaining}s`}
                  </span>
                </div>

                <Button
                  onClick={toggleSystem}
                  variant="outline"
                  className="w-full flex items-center gap-2 bg-transparent"
                  disabled={!systemStarted}
                >
                  {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                  {!systemStarted ? "Upload File First" : isRunning ? "Pause System" : "Start System"}
                </Button>
              </CardContent>
            </Card>
          </div>

          {/* Traffic Light Section */}
          <div className="flex justify-center items-start">
            <TrafficLight state={trafficState} timeRemaining={timeRemaining} ambulanceDetected={ambulanceDetected} />
          </div>
        </div>
      </div>
    </div>
  )
}
