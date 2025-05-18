'use client';
import React, { useState, useRef, useEffect } from 'react';
import { Button } from "@/components/ui/button";
import { 
  AlertCircle, 
  Camera as CameraIcon, 
  X, 
  Maximize2, 
  Minimize2,
  Move,
  Zap,
  ZapOff,
  WifiOff
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { io, Socket } from 'socket.io-client';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

interface CameraProps {
  onStateChange?: (isActive: boolean) => void;
  initialShowSetup?: boolean;
  onSetupClose?: () => void;
  onPersonRecognized?: (person: any) => void;
}

const Camera: React.FC<CameraProps> = ({ 
  onStateChange = () => {}, 
  initialShowSetup = false,
  onSetupClose = () => {},
  onPersonRecognized = () => {}
}) => {
  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const setupModalRef = useRef<HTMLDivElement>(null);
  const cameraContainerRef = useRef<HTMLDivElement>(null);
  
  // Socket.io refs
  const socketRef = useRef<Socket | null>(null);
  const streamIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const clientIdRef = useRef<string>(Date.now().toString());
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const lastFrameTimeRef = useRef(0);
  const frameBudgetRef = useRef(100); // 100ms (~10fps) initial target
  
  // State
  const [isStreaming, setIsStreaming] = useState(false);
  const [availableCameras, setAvailableCameras] = useState<MediaDeviceInfo[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [permissionState, setPermissionState] = useState<'granted' | 'denied' | 'prompt' | 'unknown'>('unknown');
  const [isExpanded, setIsExpanded] = useState(false);
  const [showCameraSetup, setShowCameraSetup] = useState(initialShowSetup);
  const [isVideoElementReady, setIsVideoElementReady] = useState(false);
  const [pendingStream, setPendingStream] = useState<MediaStream | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [position, setPosition] = useState({ x: 20, y: 20 });
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  
  // Socket.io states
  const [isProcessingEnabled, setIsProcessingEnabled] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [frameRate, setFrameRate] = useState(10); // Frames per second
  
  // Enrollment states
  const [enrollmentState, setEnrollmentState] = useState<{
    active: boolean;
    progress: number;
    facesCollected: number;
    facesNeeded: number;
    showNameDialog: boolean;
  }>({
    active: false,
    progress: 0,
    facesCollected: 0,
    facesNeeded: 5,
    showNameDialog: false,
  });
  const [newPersonName, setNewPersonName] = useState("");
  const [newPersonRelationship, setNewPersonRelationship] = useState("");

  const { toast } = useToast();

  // VideoRef setup - keep your existing code
  useEffect(() => {
    setIsVideoElementReady(videoRef.current !== null);
    
    if (pendingStream && videoRef.current && isVideoElementReady) {
      videoRef.current.srcObject = pendingStream;
      videoRef.current.play().catch(err => {
        console.error("Failed to play video with pending stream:", err);
        setError(`Could not play video: ${err.message}`);
      });
      setPendingStream(null);
    }
  }, [videoRef.current, pendingStream, isVideoElementReady]);
  
  // Error handling - keep your existing code
  useEffect(() => {
    if (error) {
      toast({
        title: "Camera Error",
        description: error,
        variant: "destructive",
        duration: 5000,
      });
      setTimeout(() => setError(null), 100);
    }
  }, [error, toast]);

  // Sync with parent's initialShowSetup - keep your existing code
  useEffect(() => {
    setShowCameraSetup(initialShowSetup);
  }, [initialShowSetup]);

  // Keep your existing drag functionality
  useEffect(() => {
    if (!isDragging) return;
    
    // Existing drag code...
    let animationFrameId: number;
    let currentX = position.x;
    let currentY = position.y;
    
    const handleMouseMove = (e: MouseEvent) => {
      e.preventDefault();
      
      // Use clientX/Y for mouse events
      const newX = e.clientX - dragOffset.x;
      const newY = e.clientY - dragOffset.y;
      
      // Update local variables immediately for smooth animation
      currentX = newX;
      currentY = newY;
      
      // Schedule update with animation frame
      if (!animationFrameId) {
        animationFrameId = requestAnimationFrame(updatePosition);
      }
    };
    
    const handleTouchMove = (e: TouchEvent) => {
      e.preventDefault();
      
      if (e.touches.length > 0) {
        const touch = e.touches[0];
        
        // Use clientX/Y for touch events
        const newX = touch.clientX - dragOffset.x;
        const newY = touch.clientY - dragOffset.y;
        
        // Update local variables immediately for smooth animation
        currentX = newX;
        currentY = newY;
        
        // Schedule update with animation frame
        if (!animationFrameId) {
          animationFrameId = requestAnimationFrame(updatePosition);
        }
      }
    };
    
    // Update function runs at screen refresh rate
    const updatePosition = () => {
      animationFrameId = 0;
      
      // Apply boundary constraints
      if (cameraContainerRef.current) {
        const containerWidth = cameraContainerRef.current.offsetWidth;
        const containerHeight = cameraContainerRef.current.offsetHeight;
        
        // Keep camera within viewport bounds
        const boundedX = Math.max(0, Math.min(window.innerWidth - containerWidth, currentX));
        const boundedY = Math.max(0, Math.min(window.innerHeight - containerHeight, currentY));
        
        setPosition({ x: boundedX, y: boundedY });
      }
    };
    
    const handleDragEnd = () => {
      setIsDragging(false);
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
    
    // Add optimized passive:false for touch events to prevent scrolling
    window.addEventListener('mousemove', handleMouseMove, { passive: false });
    window.addEventListener('touchmove', handleTouchMove, { passive: false });
    window.addEventListener('mouseup', handleDragEnd);
    window.addEventListener('touchend', handleDragEnd);
    
    // Clean up
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('touchmove', handleTouchMove);
      window.removeEventListener('mouseup', handleDragEnd);
      window.removeEventListener('touchend', handleDragEnd);
      
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, [isDragging, dragOffset]);

  // Socket.io connection effect - replace your WebRTC effect
  useEffect(() => {
    if (!isProcessingEnabled || !isStreaming) {
      // Close any existing connections when disabled
      cleanupSocketIO();
      return;
    }
    
    // Establish Socket.io connection when enabled
    const connectSocketIO = async () => {
      try {
        cleanupSocketIO();
        
        // Set up Socket.io connection
        const socketHost = window.location.hostname;
        const socketPort = "8000"; // Your backend port
        const socketURL = `http://${socketHost}:${socketPort}`;
        
        console.log(`Connecting to Socket.IO: ${socketURL}`);
        
        const socket = io(socketURL, {
          reconnectionAttempts: 5,
          reconnectionDelay: 1000,
          timeout: 10000,
          transports: ['websocket', 'polling']
        });
        
        socketRef.current = socket;
        
        // Socket.io event handlers
        socket.on('connect', () => {
          console.log('Socket.IO connection established');
          setIsConnected(true);
          
          // Register as a broadcaster
          socket.emit('register_broadcaster', {
            config: {
              quality: 'high',
              frameRate: 25
            }
          });
          
          toast({
            title: "Connected to Server",
            description: "Live video streaming active",
            duration: 3000,
          });
          
          startStreaming();
        });
        
        socket.on('disconnect', () => {
          console.log('Socket.IO disconnected');
          setIsConnected(false);
          stopStreaming();
          
          toast({
            title: "Disconnected",
            description: "Server connection lost",
            duration: 3000,
          });
        });
        
        socket.on('connection_status', (data) => {
          console.log('Connection status:', data);
          if (data.config) {
            // Apply any server configuration
            console.log('Server config:', data.config);
          }
        });
        
        socket.on('recognition_result', (data) => {
          try {
            if (data.results) {
              // Handle recognition results
              data.results.forEach(person => {
                if (person) {
                  // Pass the detected person data to parent component
                  onPersonRecognized(person);
                  
                  toast({
                    title: `Recognized: ${person.name}`,
                    description: `Confidence: ${Math.round(person.confidence * 100)}%`,
                    duration: 3000,
                  });
                }
              });
            }
          } catch (err) {
            console.error('Error processing recognition result:', err);
          }
        });
        
        // Add new event handlers here
        socket.on('new_person_detected', async (data) => {
          console.log('New person detected:', data);
          
          // Show UI for adding a new person
          toast({
            title: "New Person Detected",
            description: "Would you like to add them to your contacts?",
            action: (
              <div className="flex space-x-2">
                <Button variant="default" size="sm" onClick={() => {
                  // Open a dialog to enter person details
                  // You might want to replace this with a more elegant UI component
                  const name = prompt("Enter name:");
                  const relationship = prompt("Enter relationship:");
                  const description = prompt("Enter description:");
                  
                  if (name) {
                    socket.emit('save_new_person', {
                      face_id: data.face_id,
                      name,
                      relationship,
                      description
                    }, (response) => {
                      if (response.status === 'success') {
                        toast({
                          title: "Person Added",
                          description: `${name} was added to your contacts.`,
                          duration: 3000,
                        });
                      }
                    });
                  }
                }}>
                  Add
                </Button>
                <Button variant="outline" size="sm">
                  Skip
                </Button>
              </div>
            ),
            duration: 10000,
          });
        });
        
        socket.on('person_recognized', (data) => {
          console.log('Person recognized:', data);
          
          const person = data.person;
          
          // Pass the person data to the parent component using the existing callback
          if (person && person.name) {
            onPersonRecognized({
              id: person.id || String(Date.now()),
              name: person.name,
              relationship: person.relationship || "Unknown",
              lastMet: person.lastMet || "Just now",
              description: person.description || "",
              imageUrl: person.imageUrl ? `http://${window.location.hostname}:8000${person.imageUrl}` : "/placeholder-profile.jpg"
            });
          }
          
          // Show a notification with image if available
          toast({
            title: `Recognized: ${person.name}`,
            description: (
              <div className="flex items-center">
                {person.imageUrl && (
                  <img 
                    src={`http://${window.location.hostname}:8000${person.imageUrl}`} 
                    alt={person.name}
                    className="w-10 h-10 rounded-full object-cover mr-2"
                  />
                )}
                <span>{person.relationship || ''}. Last seen: {person.lastMet || 'Just now'}</span>
              </div>
            ),
            duration: 5000,
          });
        });
        
        socket.on('recognition_results', (data) => {
          console.log('Recognition results:', data);
          // You could use this to draw bounding boxes on a canvas overlay
          // if you want to show them on the local video feed
        });
        
        socket.on('pong', (data) => {
          console.log('Received pong:', data);
        });
        
        socket.on('error', (error) => {
          console.error('Socket.IO error:', error);
          setError(`Connection error: ${error}`);
        });
        
        // New enrollment event handlers
        socket.on('enrollment_progress', (data) => {
          console.log('Enrollment progress:', data);
          setEnrollmentState(prev => ({
            ...prev,
            active: true,
            progress: data.progress || 0,
            facesCollected: data.faces_collected || 0,
            facesNeeded: data.faces_needed || 5,
          }));
          
          toast({
            title: "Collecting faces",
            description: `${data.faces_collected}/${data.faces_needed} faces collected`,
            duration: 2000,
          });
        });
        
        socket.on('enrollment_ready', (data) => {
          console.log('Enrollment ready:', data);
          setEnrollmentState(prev => ({
            ...prev,
            active: true,
            showNameDialog: true,
            facesCollected: data.faces_collected || 5,
          }));
          
          toast({
            title: "Face Collection Complete",
            description: "Please enter a name for this person",
            duration: 5000,
          });
        });
        
        socket.on('enrollment_complete', (data) => {
          console.log('Enrollment complete:', data);
          setEnrollmentState({
            active: false,
            progress: 0,
            facesCollected: 0,
            facesNeeded: 5,
            showNameDialog: false,
          });
          
          toast({
            title: "Success!",
            description: `Added ${data.name} to your contacts`,
            duration: 3000,
          });
        });
      } catch (err) {
        console.error('Error establishing Socket.IO connection:', err);
        setError(`Socket.IO connection error: ${(err as Error).message}`);
        setIsConnected(false);
      }
    };
    
    connectSocketIO();
    
    // Cleanup function
    return () => {
      cleanupSocketIO();
    };
  }, [isProcessingEnabled, isStreaming]);

  // Clean up Socket.io connections
  const cleanupSocketIO = () => {
    // Stop streaming frames
    stopStreaming();
    
    // Disconnect Socket.io
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
    }
    
    setIsConnected(false);
  };

  // Start streaming frames to the server
  const startStreaming = () => {
    // Stop any existing interval
    stopStreaming();
    
    console.log("Starting high-quality streaming...");
    
    // Initialize timing variables
    lastFrameTimeRef.current = performance.now();
    let frameCount = 0;
    let lastFpsUpdate = performance.now();
    const targetFpsInterval = 1000 / 25; // Target 25 FPS
    
    const streamFrame = (timestamp: number) => {
      if (!socketRef.current) {
        console.log("Socket.io not available, stopping stream");
        return;
      }
      
      const elapsed = timestamp - lastFrameTimeRef.current;
      
      // Only capture if enough time has passed and not processing
      if (elapsed >= targetFpsInterval && !isProcessing) {
        captureAndSendFrame();
        frameCount++;
      }
      
      // Calculate FPS every second
      if (timestamp - lastFpsUpdate >= 1000) {
        const fps = Math.round((frameCount * 1000) / (timestamp - lastFpsUpdate));
        setFrameRate(Math.min(fps, 60)); // Cap at 60 FPS for display
        frameCount = 0;
        lastFpsUpdate = timestamp;
        
        // Log connection status for debugging
        console.log(`Streaming at ${fps} FPS, Socket.io state: ${socketRef.current?.connected}`);
      }
      
      // Schedule next frame - critical for continuous streaming
      streamIntervalRef.current = requestAnimationFrame(streamFrame);
    };
    
    // Start the animation loop
    streamIntervalRef.current = requestAnimationFrame(streamFrame);
  };

  // Stop streaming frames
  const stopStreaming = () => {
    if (typeof streamIntervalRef.current === 'number') {
      cancelAnimationFrame(streamIntervalRef.current);
      streamIntervalRef.current = null;
    }
  };

  // Replace your captureAndSendFrame function with this improved version
const captureAndSendFrame = async () => {
  if (!videoRef.current || !socketRef.current || !socketRef.current.connected) {
    return;
  }
  
  try {
    setIsProcessing(true);
    
    const video = videoRef.current;
    
    // Create a canvas to capture the frame
    const canvas = document.createElement('canvas');
    canvas.width = 640;  // Fixed width for consistent performance
    canvas.height = 480; // Fixed height for consistent performance
    
    const ctx = canvas.getContext('2d', { alpha: false });
    if (!ctx) return;
    
    // Draw the current video frame
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert to JPEG at medium quality for better performance
    const imageData = canvas.toDataURL('image/jpeg', 0.7);
    const base64Data = imageData.split('data:image/jpeg;base64,')[1];
    
    if (!base64Data) {
      console.error("Failed to get base64 data");
      return;
    }
    
    // Create frame packet
    const framePacket = {
      frame: base64Data,
      timestamp: Date.now()
    };
    
    // Send frame using volatile (can be dropped if connection is slow)
    socketRef.current.volatile.emit('frame', framePacket, (response) => {
      if (response && response.status === 'error') {
        console.error('Error sending frame:', response.message);
      }
    });
    
    // Debug log occasionally
    if (Math.random() < 0.05) { // Log only 5% of frames to avoid console spam
      console.log(`Frame sent, active viewers: ${response?.viewers || 'unknown'}`);
    }
  } catch (err) {
    console.error('Error capturing frame:', err);
  } finally {
    setIsProcessing(false);
  }
};

  // Toggle online/offline mode
  const toggleOnlineMode = () => {
    if (isConnected) {
      // Disconnect from server
      setIsProcessingEnabled(false);
      toast({
        title: "Disconnected",
        description: "Video streaming stopped",
        duration: 2000,
      });
    } else {
      // Connect to server
      setIsProcessingEnabled(true);
      toast({
        title: "Connecting...",
        description: "Starting video stream to server",
        duration: 2000,
      });
    }
  };

  // Manual frame capture for offline mode
  const captureAndProcessFrame = async () => {
    if (!videoRef.current || isProcessing) return;
    
    try {
      setIsProcessing(true);
      
      // Create a canvas to capture the frame
      const canvas = document.createElement('canvas');
      const video = videoRef.current;
      
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error("Could not get canvas context");
      
      // Draw the current video frame
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      // Convert to a data URL
      const imageData = canvas.toDataURL('image/jpeg', 0.8);
      
      // Show processing indicator
      toast({
        title: "Processing...",
        description: "Analyzing the image",
        duration: 2000,
      });
      
      // Send for processing via fetch API
      const response = await fetch('/api/recognize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData }),
      });
      
      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }
      
      const result = await response.json();
      
      if (result.results && result.results.length > 0) {
        const person = result.results[0];
        onPersonRecognized(person);
        
        toast({
          title: `Recognized: ${person.name}`,
          description: `Confidence: ${Math.round(person.confidence * 100)}%`,
          duration: 3000,
        });
      } else {
        toast({
          title: "No Match Found",
          description: "Could not recognize anyone in this image",
          duration: 2000,
        });
      }
    } catch (err) {
      console.error('Error processing frame:', err);
      toast({
        title: "Processing Failed",
        description: `Error: ${(err as Error).message}`,
        variant: "destructive",
        duration: 3000,
      });
    } finally {
      setTimeout(() => setIsProcessing(false), 1000);
    }
  };
  
 

  // Keep your existing camera permissions check
  useEffect(() => {
    const checkPermissions = async () => {
      try {
        if (navigator.permissions) {
          const result = await navigator.permissions.query({ name: 'camera' as PermissionName });
          setPermissionState(result.state as 'granted' | 'denied' | 'prompt');
          
          if (result.state === 'granted') {
            await getAvailableCameras();
          }
        } else {
          try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            await getAvailableCameras();
            setPermissionState('granted');
          } catch (err) {
            setPermissionState('prompt');
          }
        }
      } catch (err) {
        console.error("Error checking permissions:", err);
        setError(`Permission check failed: ${(err as Error).message}`);
      }
    };

    checkPermissions();
    
    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        stopCamera();
      }
    };
  }, []); 

  // Keep your existing drag start handler
  const handleDragStart = (e: React.MouseEvent<HTMLDivElement> | React.TouchEvent<HTMLDivElement>) => {
    e.preventDefault();
    
    if (!cameraContainerRef.current) return;
    
    const rect = cameraContainerRef.current.getBoundingClientRect();
    
    if ('touches' in e) { // Touch event
      const touch = e.touches[0];
      const offsetX = touch.clientX - rect.left;
      const offsetY = touch.clientY - rect.top;
      setDragOffset({ x: offsetX, y: offsetY });
    } else { // Mouse event  
      const offsetX = e.clientX - rect.left;
      const offsetY = e.clientY - rect.top;
      setDragOffset({ x: offsetX, y: offsetY });
    }
    
    setIsDragging(true);
  };

  // Keep your existing camera functions
  const getAvailableCameras = async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(device => device.kind === 'videoinput');
      
      setAvailableCameras(videoDevices);
      
      if (videoDevices.length > 0) {
        setSelectedCamera(videoDevices[0].deviceId);
      } else {
        setError('No cameras detected on your device');
      }
    } catch (err) {
      console.error("Error getting cameras:", err);
      setError('Unable to find cameras: ' + (err as Error).message);
    }
  };

  const requestCameraPermission = async () => {
    try {
      console.log("Requesting camera permission...");
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      
      stream.getTracks().forEach(track => track.stop());
      
      toast({
        title: "Camera access granted",
        description: "You can now use your camera.",
        duration: 3000,
      });
      
      setPermissionState('granted');
      await getAvailableCameras();
      setError(null);
    } catch (err) {
      console.error("Camera permission error:", err);
      setError('Camera permission denied. Please allow camera access to use this feature.');
      setPermissionState('denied');
    }
  };

  const startCamera = async () => {
    try {
      console.log("Starting camera...");
      setError(null);
      
      if (!selectedCamera) {
        setError('No camera selected');
        return;
      }

      // First, update UI state to render the video element
      setIsStreaming(true);
      setShowCameraSetup(false);
      onSetupClose();
      onStateChange(true);
      
      // Get the media stream
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          deviceId: { exact: selectedCamera },
          width: { ideal: 1280 },
          height: { ideal: 720 }
        }
      });
      
      // Set as pending stream if video element not ready yet
      if (!videoRef.current || !isVideoElementReady) {
        console.log("Video element not ready yet, setting pending stream");
        setPendingStream(stream);
        return;
      }
      
      // If we have the video element, attach stream directly
      console.log("Video element ready, attaching stream");
      videoRef.current.srcObject = stream;
      
      try {
        await videoRef.current.play();
        console.log("Video playback started successfully");
        
        toast({
          title: "Camera active",
          description: "Drag the camera to move it around the screen.",
          duration: 3000,
        });
      } catch (playError) {
        console.error("Failed to play video:", playError);
        setError(`Could not start video: ${(playError as Error).message}`);
      }
    } catch (err) {
      console.error("Camera start error:", err);
      setError('Failed to access camera: ' + (err as Error).message);
      setIsStreaming(false);
      onStateChange(false);
    }
  };

  const stopCamera = () => {
    console.log("Stopping camera...");
    
    // Clean up Socket.io connections first
    setIsProcessingEnabled(false);
    cleanupSocketIO();
    
    if (videoRef.current && videoRef.current.srcObject) {
      try {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
        tracks.forEach(track => track.stop());
        videoRef.current.srcObject = null;
      } catch (err) {
        console.error("Error stopping camera:", err);
      }
    }
    
    // Clear any pending stream
    if (pendingStream) {
      pendingStream.getTracks().forEach(track => track.stop());
      setPendingStream(null);
    }
    
    setIsStreaming(false);
    onStateChange(false);
    
    toast({
      title: "Camera stopped",
      description: "Your camera has been turned off.",
      duration: 3000,
    });
  };

  const handleCameraChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = e.target.value;
    setSelectedCamera(value);
    
    if (isStreaming) {
      stopCamera();
      setTimeout(() => {
        startCamera();
      }, 500);
    }
  };

  const toggleCameraExpanded = () => {
    setIsExpanded(!isExpanded);
  };

  const closeSetup = () => {
    setShowCameraSetup(false);
    onSetupClose();
  };

  // Keep your existing renderCameraSetup function

  const renderCameraSetup = () => {
    if (!showCameraSetup) return null;
    
    return (
      <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
        <div 
          ref={setupModalRef}
          className="bg-slate-800 text-white rounded-2xl shadow-xl max-w-sm w-full overflow-hidden border border-slate-700"
        >
          {/* Header */}
          <div className="bg-indigo-700 text-white px-5 py-4 flex justify-between items-center">
            <h3 className="text-lg font-semibold flex items-center">
              <CameraIcon className="mr-2 h-5 w-5" /> Camera Setup
            </h3>
            <button 
              onClick={closeSetup}
              className="text-white/80 hover:text-white focus:outline-none focus:ring-2 focus:ring-white/50 rounded-full"
              aria-label="Close"
            >
              <X size={20} />
            </button>
          </div>
          
          <div className="p-5">
            {/* Camera permission states - keep your existing code */}
            {permissionState === 'denied' && (
              <div className="bg-yellow-900/30 border border-yellow-700 rounded-lg p-4 mb-4">
                <div className="flex items-start">
                  <AlertCircle className="h-5 w-5 text-yellow-400 mt-0.5 mr-2" />
                  <div>
                    <h4 className="font-medium text-yellow-400">Camera Permission Required</h4>
                    <p className="text-yellow-200/80 text-sm mt-1">
                      Please update your browser settings to allow camera access.
                    </p>
                  </div>
                </div>
              </div>
            )}
            
            {permissionState === 'prompt' && (
              <div className="text-center p-4">
                <CameraIcon className="h-12 w-12 text-indigo-400 mx-auto mb-3" />
                <h3 className="font-medium text-lg text-white mb-2">Camera Permission Needed</h3>
                <p className="text-slate-300 mb-6">
                  We need camera access to help you recognize people around you.
                </p>
                <Button 
                  onClick={requestCameraPermission}
                  className="bg-indigo-600 hover:bg-indigo-700 text-white w-full py-5"
                >
                  Allow Camera Access
                </Button>
              </div>
            )}
            
            {permissionState === 'granted' && (
              <>
                <div className="mb-5">
                  <label className="block text-sm font-medium text-slate-200 mb-2">
                    Select Camera
                  </label>
                  <select 
                    className="w-full p-2.5 border bg-slate-700 text-white border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    value={selectedCamera}
                    onChange={handleCameraChange}
                  >
                    {availableCameras.length === 0 ? (
                      <option value="">No cameras found</option>
                    ) : (
                      availableCameras.map(camera => (
                        <option key={camera.deviceId} value={camera.deviceId}>
                          {camera.label || `Camera ${camera.deviceId.substring(0, 5)}...`}
                        </option>
                      )))
                    }
                  </select>
                  
                  <p className="text-sm text-slate-400 mt-3">
                    The camera will appear as a floating window that you can drag around the screen.
                  </p>
                </div>
                
                <div className="flex justify-end space-x-3 mt-6">
                  <Button 
                    variant="outline"
                    onClick={closeSetup}
                    className="border-slate-600 text-slate-300 hover:bg-slate-700 hover:text-white"
                  >
                    Cancel
                  </Button>
                  
                  <Button 
                    onClick={startCamera}
                    disabled={!selectedCamera}
                    className="bg-indigo-600 hover:bg-indigo-700 text-white"
                  >
                    <CameraIcon className="mr-2 h-4 w-4" /> Start Camera
                  </Button>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    );
  };

  // Update the renderFixedCornerCamera function with WebRTC controls
  const renderFixedCornerCamera = () => {
    // Always render if streaming, even if we're waiting for the video ref
    const size = isExpanded 
      ? { width: '320px', height: '240px' } 
      : { width: '180px', height: '135px' };
    
    return (
      <div
        ref={cameraContainerRef}
        style={{ 
          width: size.width,
          height: size.height,
          left: `${position.x}px`,
          top: `${position.y}px`,
          transform: 'translate3d(0,0,0)' // Force hardware acceleration
        }}
        className="fixed z-[9999] bg-black rounded-2xl shadow-2xl overflow-hidden border-2 border-indigo-500 will-change-transform"
      >
        {/* Video element */}
        <video 
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-cover"
          style={{ 
            objectFit: 'cover',
            transform: 'translate3d(0,0,0)', // Force hardware acceleration
            willChange: 'transform', // Hint to browser
            cursor: !isProcessingEnabled ? 'pointer' : 'default'
          }}
          onClick={!isProcessingEnabled ? captureAndProcessFrame : undefined}
        />
        
        {/* Drag handle indicator */}
        <div 
          className="absolute top-0 left-0 right-0 h-12 bg-gradient-to-b from-black/50 to-transparent z-20 cursor-grab active:cursor-grabbing flex items-center justify-center touch-none"
          onMouseDown={handleDragStart}
          onTouchStart={handleDragStart}
        >
          <div className="w-16 h-1.5 bg-white/40 rounded-full"></div>
          <Move size={16} className="absolute right-3 top-3 text-white/60" />
          
          {/* Connection status indicator */}
          <div className={`absolute left-3 top-3 flex items-center ${isConnected ? 'text-green-400' : 'text-gray-400'}`}>
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400' : 'bg-gray-400'} ${isProcessing ? 'animate-pulse' : ''} mr-1`}></div>
            <span className="text-xs">{isConnected ? 'Connected' : 'Offline'}</span>
          </div>
        </div>

        {/* Processing overlay - show when processing a frame */}
        {isProcessing && (
          <div className="absolute inset-0 bg-black/30 backdrop-blur-sm flex items-center justify-center z-30">
            <div className="bg-black/70 rounded-full p-3">
              <div className="h-8 w-8 border-2 border-t-indigo-500 border-r-indigo-500 border-indigo-500/30 rounded-full animate-spin"></div>
            </div>
          </div>
        )}
        
        {/* Controls overlay */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent opacity-0 hover:opacity-100 focus-within:opacity-100 active:opacity-100 transition-opacity duration-200 flex flex-col justify-between p-3">
          {/* Top controls */}
          <div className="flex justify-end space-x-3 pt-8">
            {/* Online/Offline toggle */}
            <button 
              onClick={toggleOnlineMode}
              className={`${
                isConnected 
                  ? "bg-green-600 hover:bg-green-700 active:bg-green-800" 
                  : "bg-slate-600 hover:bg-slate-700 active:bg-slate-800"
              } text-white p-3 rounded-full active:scale-95 transition-all duration-75 touch-manipulation flex items-center justify-center relative`}
              aria-label={isConnected ? "Go offline" : "Connect to server"}
            >
              {isConnected ? (
                <>
                  <Zap size={18} />
                  <span className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full animate-pulse border border-white"></span>
                </>
              ) : (
                <WifiOff size={18} />
              )}
            </button>
            
            {/* Size toggle button */}
            <button 
              onClick={toggleCameraExpanded}
              className="bg-white text-black p-3 rounded-full hover:bg-gray-200 active:bg-gray-300 active:scale-95 transition-all duration-75 touch-manipulation"
              aria-label={isExpanded ? "Shrink camera" : "Expand camera"}
            >
              {isExpanded ? <Minimize2 size={18} /> : <Maximize2 size={18} />}
            </button>
            
            {/* Close button */}
            <button 
              onClick={stopCamera}
              className="bg-red-600 text-white p-3 rounded-full hover:bg-red-700 active:bg-red-800 active:scale-95 transition-all duration-75 touch-manipulation"
              aria-label="Stop camera"
            >
              <X size={18} />
            </button>
          </div>
          
          {/* Bottom status indicator */}
          <div className="self-start bg-black/70 backdrop-blur-sm px-3 py-2 rounded-full flex items-center">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400' : 'bg-gray-400'} ${isProcessing ? 'animate-pulse' : ''} mr-1.5`}></div>
            <span className="text-white text-xs font-medium">
              {isConnected ? "Online" : "Offline"}
            </span>
            {isConnected && (
              <span className="text-xs text-green-200 ml-1.5">
                {Math.round(1000 / frameBudgetRef.current)} FPS
              </span>
            )}
          </div>
        </div>
      </div>
    );
  };
  
  // Always render an invisible video element to establish the ref
  const renderHiddenVideoElement = () => {
    if (isStreaming) return null; // Already rendering in the main UI
    
    return (
      <video 
        ref={videoRef}
        style={{ width: 0, height: 0, position: 'absolute', left: -9999, opacity: 0 }}
        autoPlay
        playsInline
        muted
      />
    );
  };

  // 3. Add a debug component to show the camera feed locally
const renderDebugView = () => {
  if (!isStreaming) return null;
  
  return (
    <div className="fixed bottom-4 left-4 p-2 bg-black/80 rounded-lg z-50">
      <div className="text-xs text-white mb-1">Debug View</div>
      <div className="relative w-[160px] h-[120px] overflow-hidden border border-white/20">
        <video 
          ref={(video) => {
            if (!video || !videoRef.current?.srcObject) return;
            // Clone the stream for debug view to avoid interfering with main video
            if (!video.srcObject && videoRef.current.srcObject) {
              video.srcObject = videoRef.current.srcObject;
              video.play().catch(e => console.error("Debug video play error:", e));
            }
          }}
          autoPlay
          muted
          playsInline
          className="w-full h-full object-cover"
        />
        <div className="absolute bottom-1 right-1 text-xs bg-black/60 text-white px-1 rounded">
          {frameRate} FPS
        </div>
      </div>
      <div className="text-xs text-white mt-1 flex justify-between">
        <span>
          WS: {isConnected ? '🟢 Connected' : '🔴 Disconnected'}
        </span>
        <span>
          {isProcessing ? '⚡ Processing' : '✓ Ready'}
        </span>
      </div>
    </div>
  );
};

// Add a function to handle saving the person
const handleSavePerson = () => {
  if (!newPersonName.trim()) {
    toast({
      title: "Name Required",
      description: "Please enter a name for this person",
      variant: "destructive",
      duration: 3000,
    });
    return;
  }
  
  socketRef.current?.emit('save_enrolled_face', {
    name: newPersonName.trim(),
    relationship: newPersonRelationship.trim(),
    description: "Added via auto-enrollment"
  }, (response) => {
    if (response.status === 'success') {
      setNewPersonName("");
      setNewPersonRelationship("");
      setEnrollmentState(prev => ({
        ...prev,
        showNameDialog: false,
      }));
    } else {
      toast({
        title: "Error",
        description: response.message || "Failed to save person",
        variant: "destructive",
        duration: 3000,
      });
    }
  });
};

// Add the name dialog component
const renderNameDialog = () => {
  if (!enrollmentState.showNameDialog) return null;
  
  return (
    <Dialog open={enrollmentState.showNameDialog} onOpenChange={(open) => {
      if (!open) {
        // Allow closing only if we've entered a name or explicitly cancel
        if (newPersonName.trim()) {
          setEnrollmentState(prev => ({...prev, showNameDialog: false}));
        } else {
          toast({
            title: "Name Required",
            description: "Please enter a name or cancel enrollment",
            variant: "destructive",
            duration: 3000,
          });
        }
      }
    }}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>New Person Detected</DialogTitle>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="name" className="text-right">Name</Label>
            <Input 
              id="name" 
              placeholder="Enter name" 
              className="col-span-3"
              value={newPersonName}
              onChange={(e) => setNewPersonName(e.target.value)}
              autoFocus
            />
          </div>
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="relationship" className="text-right">Relationship</Label>
            <Input 
              id="relationship" 
              placeholder="Friend, Family, etc." 
              className="col-span-3"
              value={newPersonRelationship}
              onChange={(e) => setNewPersonRelationship(e.target.value)}
            />
          </div>
        </div>
        <DialogFooter>
          <Button 
            variant="outline" 
            onClick={() => {
              // Cancel enrollment
              socketRef.current?.emit('cancel_face_collection');
              setEnrollmentState({
                active: false,
                progress: 0,
                facesCollected: 0,
                facesNeeded: 5,
                showNameDialog: false,
              });
            }}
          >
            Cancel
          </Button>
          <Button onClick={handleSavePerson}>Save Person</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

// Add progress indicator when collecting faces
const renderEnrollmentProgress = () => {
  if (!enrollmentState.active || enrollmentState.showNameDialog) return null;
  
  return (
    <div className="absolute inset-x-0 bottom-0 bg-black/50 p-2 z-20">
      <div className="text-white text-xs font-medium mb-1">
        Collecting face data: {enrollmentState.facesCollected}/{enrollmentState.facesNeeded}
      </div>
      <div className="w-full h-1.5 bg-gray-600 rounded-full overflow-hidden">
        <div 
          className="h-full bg-blue-500 rounded-full transition-all duration-300"
          style={{ width: `${enrollmentState.progress}%` }}
        ></div>
      </div>
    </div>
  );
};

  return (
    <>
      {renderHiddenVideoElement()}
      {showCameraSetup && renderCameraSetup()}
      {isStreaming && renderFixedCornerCamera()}
      {isStreaming && renderDebugView()}
      {renderNameDialog()}
      
      {/* Add the enrollment progress inside your camera container */}
      {/* You can add this to your renderFixedCornerCamera function */}
      {/* {enrollmentState.active && renderEnrollmentProgress()} */}
    </>
  );
};

export default Camera;





















































































































































// // Backup code



// 'use client';
// import React, { useState, useRef, useEffect } from 'react';
// import { Button } from "@/components/ui/button";
// import { 
//   AlertCircle, 
//   Camera as CameraIcon, 
//   X, 
//   Maximize2, 
//   Minimize2,
//   Move
// } from "lucide-react";
// import { useToast } from "@/hooks/use-toast";

// interface CameraProps {
//   onStateChange?: (isActive: boolean) => void;
//   initialShowSetup?: boolean;
//   onSetupClose?: () => void;
// }

// const Camera: React.FC<CameraProps> = ({ 
//   onStateChange = () => {}, 
//   initialShowSetup = false,
//   onSetupClose = () => {}
// }) => {
//   // Refs
//   const videoRef = useRef<HTMLVideoElement>(null);
//   const setupModalRef = useRef<HTMLDivElement>(null);
//   const cameraContainerRef = useRef<HTMLDivElement>(null);
  
//   // State
//   const [isStreaming, setIsStreaming] = useState(false);
//   const [availableCameras, setAvailableCameras] = useState<MediaDeviceInfo[]>([]);
//   const [selectedCamera, setSelectedCamera] = useState<string>('');
//   const [error, setError] = useState<string | null>(null);
//   const [permissionState, setPermissionState] = useState<'granted' | 'denied' | 'prompt' | 'unknown'>('unknown');
//   const [isExpanded, setIsExpanded] = useState(false);
//   const [showCameraSetup, setShowCameraSetup] = useState(initialShowSetup);
//   const [isVideoElementReady, setIsVideoElementReady] = useState(false);
//   const [pendingStream, setPendingStream] = useState<MediaStream | null>(null);
//   const [isDragging, setIsDragging] = useState(false);
//   const [position, setPosition] = useState({ x: 20, y: 20 });
//   const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
//   const { toast } = useToast();

//   // Always render video element to establish ref
//   useEffect(() => {
//     setIsVideoElementReady(videoRef.current !== null);
    
//     // If we have a pending stream and video element is ready, attach it
//     if (pendingStream && videoRef.current && isVideoElementReady) {
//       videoRef.current.srcObject = pendingStream;
//       videoRef.current.play().catch(err => {
//         console.error("Failed to play video with pending stream:", err);
//         setError(`Could not play video: ${err.message}`);
//       });
//       setPendingStream(null);
//     }
//   }, [videoRef.current, pendingStream, isVideoElementReady]);
  
//   // Display errors using toast instead of inline alerts
//   useEffect(() => {
//     if (error) {
//       toast({
//         title: "Camera Error",
//         description: error,
//         variant: "destructive",
//         duration: 5000,
//       });
//       setTimeout(() => setError(null), 100);
//     }
//   }, [error, toast]);

//   // Sync with parent's initialShowSetup
//   useEffect(() => {
//     setShowCameraSetup(initialShowSetup);
//   }, [initialShowSetup]);

//   // Optimized drag functionality with requestAnimationFrame
//   useEffect(() => {
//     if (!isDragging) return;
    
//     let animationFrameId: number;
//     let currentX = position.x;
//     let currentY = position.y;
    
//     const handleMouseMove = (e: MouseEvent) => {
//       e.preventDefault();
      
//       // Use clientX/Y for mouse events
//       const newX = e.clientX - dragOffset.x;
//       const newY = e.clientY - dragOffset.y;
      
//       // Update local variables immediately for smooth animation
//       currentX = newX;
//       currentY = newY;
      
//       // Schedule update with animation frame
//       if (!animationFrameId) {
    //       animationFrameId = requestAnimationFrame(updatePosition);
    //     }
    //   };
    
    //   const handleTouchMove = (e: TouchEvent) => {
    //     e.preventDefault();
      
    //     if (e.touches.length > 0) {
    //       const touch = e.touches[0];
        
    //       // Use clientX/Y for touch events
    //       const newX = touch.clientX - dragOffset.x;
    //       const newY = touch.clientY - dragOffset.y;
        
    //       // Update local variables immediately for smooth animation
    //       currentX = newX;
    //       currentY = newY;
        
    //       // Schedule update with animation frame
    //       if (!animationFrameId) {
    //         animationFrameId = requestAnimationFrame(updatePosition);
    //       }
    //     }
    //   };
    
    //   // Update function runs at screen refresh rate
    //   const updatePosition = () => {
    //     animationFrameId = 0;
      
    //     // Apply boundary constraints
    //     if (cameraContainerRef.current) {
    //       const containerWidth = cameraContainerRef.current.offsetWidth;
    //       const containerHeight = cameraContainerRef.current.offsetHeight;
        
    //       // Keep camera within viewport bounds
    //       const boundedX = Math.max(0, Math.min(window.innerWidth - containerWidth, currentX));
    //       const boundedY = Math.max(0, Math.min(window.innerHeight - containerHeight, currentY));
        
    //       setPosition({ x: boundedX, y: boundedY });
    //     }
    //   };
    
    //   const handleDragEnd = () => {
    //     setIsDragging(false);
    //     if (animationFrameId) {
    //       cancelAnimationFrame(animationFrameId);
    //     }
    //   };
    
    //   // Add optimized passive:false for touch events to prevent scrolling
    //   window.addEventListener('mousemove', handleMouseMove, { passive: false });
    //   window.addEventListener('touchmove', handleTouchMove, { passive: false });
    //   window.addEventListener('mouseup', handleDragEnd);
    //   window.addEventListener('touchend', handleDragEnd);
    
    //   // Clean up
    //   return () => {
    //     window.removeEventListener('mousemove', handleMouseMove);
    //     window.removeEventListener('touchmove', handleTouchMove);
    //     window.removeEventListener('mouseup', handleDragEnd);
    //     window.removeEventListener('touchend', handleDragEnd);
      
    //     if (animationFrameId) {
    //       cancelAnimationFrame(animationFrameId);
    //     }
    //   };
    // }, [isDragging, dragOffset]);

//   // Initialize camera permissions check
//   useEffect(() => {
//     const checkPermissions = async () => {
//       try {
//         if (navigator.permissions) {
//           const result = await navigator.permissions.query({ name: 'camera' as PermissionName });
//           setPermissionState(result.state as 'granted' | 'denied' | 'prompt');
          
//           if (result.state === 'granted') {
//             await getAvailableCameras();
//           }
//         } else {
//           try {
//             const devices = await navigator.mediaDevices.enumerateDevices();
//             await getAvailableCameras();
//             setPermissionState('granted');
//           } catch (err) {
//             setPermissionState('prompt');
//           }
//         }
//       } catch (err) {
//         console.error("Error checking permissions:", err);
//         setError(`Permission check failed: ${(err as Error).message}`);
//       }
//     };

//     checkPermissions();
    
//     return () => {
//       if (videoRef.current && videoRef.current.srcObject) {
//         stopCamera();
//       }
//     };
//   }, []); 

//   // Optimized drag start with offset calculation
//   const handleDragStart = (e: React.MouseEvent<HTMLDivElement> | React.TouchEvent<HTMLDivElement>) => {
//     e.preventDefault();
    
//     if (!cameraContainerRef.current) return;
    
//     const rect = cameraContainerRef.current.getBoundingClientRect();
    
//     if ('touches' in e) { // Touch event
//       const touch = e.touches[0];
//       const offsetX = touch.clientX - rect.left;
//       const offsetY = touch.clientY - rect.top;
//       setDragOffset({ x: offsetX, y: offsetY });
//     } else { // Mouse event  
//       const offsetX = e.clientX - rect.left;
//       const offsetY = e.clientY - rect.top;
//       setDragOffset({ x: offsetX, y: offsetY });
//     }
    
//     setIsDragging(true);
//   };

//   const getAvailableCameras = async () => {
//     try {
//       const devices = await navigator.mediaDevices.enumerateDevices();
//       const videoDevices = devices.filter(device => device.kind === 'videoinput');
      
//       setAvailableCameras(videoDevices);
      
//       if (videoDevices.length > 0) {
//         setSelectedCamera(videoDevices[0].deviceId);
//       } else {
//         setError('No cameras detected on your device');
//       }
//     } catch (err) {
//       console.error("Error getting cameras:", err);
//       setError('Unable to find cameras: ' + (err as Error).message);
//     }
//   };

//   const requestCameraPermission = async () => {
//     try {
//       console.log("Requesting camera permission...");
//       const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      
//       stream.getTracks().forEach(track => track.stop());
      
//       toast({
//         title: "Camera access granted",
//         description: "You can now use your camera.",
//         duration: 3000,
//       });
      
//       setPermissionState('granted');
//       await getAvailableCameras();
//       setError(null);
//     } catch (err) {
//       console.error("Camera permission error:", err);
//       setError('Camera permission denied. Please allow camera access to use this feature.');
//       setPermissionState('denied');
//     }
//   };

//   // Two-step camera start to avoid the "video ref is null" issue
//   const startCamera = async () => {
//     try {
//       console.log("Starting camera...");
//       setError(null);
      
//       if (!selectedCamera) {
//         setError('No camera selected');
//         return;
//       }

//       // First, update UI state to render the video element
//       setIsStreaming(true);
//       setShowCameraSetup(false);
//       onSetupClose();
//       onStateChange(true);
      
//       // Get the media stream
//       const stream = await navigator.mediaDevices.getUserMedia({
//         video: {
//           deviceId: { exact: selectedCamera },
//           width: { ideal: 1280 },
//           height: { ideal: 720 }
//         }
//       });
      
//       // Set as pending stream if video element not ready yet
//       if (!videoRef.current || !isVideoElementReady) {
//         console.log("Video element not ready yet, setting pending stream");
//         setPendingStream(stream);
//         return;
//       }
      
//       // If we have the video element, attach stream directly
//       console.log("Video element ready, attaching stream");
//       videoRef.current.srcObject = stream;
      
//       try {
//         await videoRef.current.play();
//         console.log("Video playback started successfully");
        
//         toast({
//           title: "Camera active",
//           description: "Drag the camera to move it around the screen.",
//           duration: 3000,
//         });
//       } catch (playError) {
//         console.error("Failed to play video:", playError);
//         setError(`Could not start video: ${(playError as Error).message}`);
//       }
//     } catch (err) {
//       console.error("Camera start error:", err);
//       setError('Failed to access camera: ' + (err as Error).message);
//       setIsStreaming(false);
//       onStateChange(false);
//     }
//   };

//   const stopCamera = () => {
//     console.log("Stopping camera...");
//     if (videoRef.current && videoRef.current.srcObject) {
//       try {
//         const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
//         tracks.forEach(track => track.stop());
//         videoRef.current.srcObject = null;
//       } catch (err) {
//         console.error("Error stopping camera:", err);
//       }
//     }
    
//     // Clear any pending stream
//     if (pendingStream) {
//       pendingStream.getTracks().forEach(track => track.stop());
//       setPendingStream(null);
//     }
    
//     setIsStreaming(false);
//     onStateChange(false);
    
//     toast({
//       title: "Camera stopped",
//       description: "Your camera has been turned off.",
//       duration: 3000,
//     });
//   };

//   const handleCameraChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
//     const value = e.target.value;
//     setSelectedCamera(value);
    
//     if (isStreaming) {
//       stopCamera();
//       setTimeout(() => {
//         startCamera();
//       }, 500);
//     }
//   };

//   const toggleCameraExpanded = () => {
//     setIsExpanded(!isExpanded);
//   };

//   const closeSetup = () => {
//     setShowCameraSetup(false);
//     onSetupClose();
//   };

//   const renderCameraSetup = () => {
//     if (!showCameraSetup) return null;
    
//     return (
//       <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
//         <div 
//           ref={setupModalRef}
//           className="bg-slate-800 text-white rounded-2xl shadow-xl max-w-sm w-full overflow-hidden border border-slate-700"
//         >
//           {/* Header */}
//           <div className="bg-indigo-700 text-white px-5 py-4 flex justify-between items-center">
//             <h3 className="text-lg font-semibold flex items-center">
//               <CameraIcon className="mr-2 h-5 w-5" /> Camera Setup
//             </h3>
//             <button 
//               onClick={closeSetup}
//               className="text-white/80 hover:text-white focus:outline-none focus:ring-2 focus:ring-white/50 rounded-full"
//               aria-label="Close"
//             >
//               <X size={20} />
//             </button>
//           </div>
          
//           <div className="p-5">
//             {permissionState === 'denied' && (
//               <div className="bg-yellow-900/30 border border-yellow-700 rounded-lg p-4 mb-4">
//                 <div className="flex items-start">
//                   <AlertCircle className="h-5 w-5 text-yellow-400 mt-0.5 mr-2" />
//                   <div>
//                     <h4 className="font-medium text-yellow-400">Camera Permission Required</h4>
//                     <p className="text-yellow-200/80 text-sm mt-1">
//                       Please update your browser settings to allow camera access.
//                     </p>
//                   </div>
//                 </div>
//               </div>
//             )}
//             
//             {permissionState === 'prompt' && (
//               <div className="text-center p-4">
//                 <CameraIcon className="h-12 w-12 text-indigo-400 mx-auto mb-3" />
//                 <h3 className="font-medium text-lg text-white mb-2">Camera Permission Needed</h3>
//                 <p className="text-slate-300 mb-6">
//                   We need camera access to help you recognize people around you.
//                 </p>
//                 <Button 
//                   onClick={requestCameraPermission}
//                   className="bg-indigo-600 hover:bg-indigo-700 text-white w-full py-5"
//                 >
//                   Allow Camera Access
//                 </Button>
//               </div>
//             )}
//             
//             {permissionState === 'granted' && (
//               <>
//                 <div className="mb-5">
//                   <label className="block text-sm font-medium text-slate-200 mb-2">
//                     Select Camera
//                   </label>
//                   <select 
//                     className="w-full p-2.5 border bg-slate-700 text-white border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
//                     value={selectedCamera}
//                     onChange={handleCameraChange}
//                   >
//                     {availableCameras.length === 0 ? (
//                       <option value="">No cameras found</option>
//                     ) : (
//                       availableCameras.map(camera => (
//                         <option key={camera.deviceId} value={camera.deviceId}>
//                           {camera.label || `Camera ${camera.deviceId.substring(0, 5)}...`}
//                         </option>
//                       )))
//                     }
//                   </select>
                  
//                   <p className="text-sm text-slate-400 mt-3">
//                     The camera will appear as a floating window that you can drag around the screen.
//                   </p>
//                 </div>
                
//                 <div className="flex justify-end space-x-3 mt-6">
//                   <Button 
//                     variant="outline"
//                     onClick={closeSetup}
//                     className="border-slate-600 text-slate-300 hover:bg-slate-700 hover:text-white"
//                   >
//                     Cancel
//                   </Button>
                  
//                   <Button 
//                     onClick={startCamera}
//                     disabled={!selectedCamera}
//                     className="bg-indigo-600 hover:bg-indigo-700 text-white"
//                   >
//                     <CameraIcon className="mr-2 h-4 w-4" /> Start Camera
//                   </Button>
//                 </div>
//               </>
//             )}
//           </div>
//         </div>
//       </div>
//     );
//   };

//   const renderFixedCornerCamera = () => {
//     // Always render if streaming, even if we're waiting for the video ref
//     const size = isExpanded 
//       ? { width: '320px', height: '240px' } 
//       : { width: '180px', height: '135px' };
    
//     return (
//       <div
//         ref={cameraContainerRef}
//         style={{ 
//           width: size.width,
//           height: size.height,
//           left: `${position.x}px`,
//           top: `${position.y}px`,
//           transform: 'translate3d(0,0,0)' // Force hardware acceleration
//         }}
//         className="fixed z-[9999] bg-black rounded-2xl shadow-2xl overflow-hidden border-2 border-indigo-500 will-change-transform"
//       >
//         {/* Video element - ALWAYS render this first for better performance */}
//         <video 
//           ref={videoRef}
//           autoPlay
//           playsInline
//           muted
//           className="w-full h-full object-cover"
//         />
        
//         {/* Drag handle indicator - improved touch target */}
//         <div 
//           className="absolute top-0 left-0 right-0 h-12 bg-gradient-to-b from-black/50 to-transparent z-20 cursor-grab active:cursor-grabbing flex items-center justify-center touch-none"
//           onMouseDown={handleDragStart}
//           onTouchStart={handleDragStart}
//         >
//           <div className="w-16 h-1.5 bg-white/40 rounded-full"></div>
//           <Move size={16} className="absolute right-3 top-3 text-white/60" />
//         </div>

//         {/* Loading indicator */}
//         {(!videoRef.current?.srcObject && !pendingStream) && (
//           <div className="absolute inset-0 flex items-center justify-center z-10 bg-black/60 text-white text-sm">
//             Camera initializing...
//           </div>
//         )}
//         
//         {/* Controls overlay */}
//         <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent opacity-0 hover:opacity-100 focus-within:opacity-100 active:opacity-100 transition-opacity duration-200 flex flex-col justify-between p-3">
//           {/* Top controls - larger touch targets */}
//           <div className="flex justify-end space-x-3 pt-8">
//             <button 
//               onClick={toggleCameraExpanded}
//               className="bg-white text-black p-3 rounded-full hover:bg-gray-200 touch-manipulation"
//               aria-label={isExpanded ? "Shrink camera" : "Expand camera"}
//             >
//               {isExpanded ? <Minimize2 size={18} /> : <Maximize2 size={18} />}
//             </button>
//             <button 
//               onClick={stopCamera}
//               className="bg-red-600 text-white p-3 rounded-full hover:bg-red-700 touch-manipulation"
//               aria-label="Stop camera"
//             >
//               <X size={18} />
//             </button>
//           </div>
          
//           {/* Bottom indicator */}
//           <div className="self-start bg-red-500 px-3 py-2 rounded-full flex items-center">
//             <div className="w-2 h-2 rounded-full bg-white animate-pulse mr-1.5"></div>
//             <span className="text-white text-xs font-medium">Live</span>
//           </div>
//         </div>
//       </div>
//     );
//   };
  
//   // Always render an invisible video element to establish the ref
//   const renderHiddenVideoElement = () => {
//     if (isStreaming) return null; // Already rendering in the main UI
    
//     return (
//       <video 
//         ref={videoRef}
//         style={{ width: 0, height: 0, position: 'absolute', left: -9999, opacity: 0 }}
//         autoPlay
//         playsInline
//         muted
//       />
//     );
//   };

//   return (
//     <>
//       {renderHiddenVideoElement()}
//       {showCameraSetup && renderCameraSetup()}
//       {isStreaming && renderFixedCornerCamera()}
//     </>
//   );
// };

// export default Camera;


