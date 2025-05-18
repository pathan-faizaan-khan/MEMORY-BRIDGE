'use client';
import { useState, useEffect, useCallback } from "react";
import Image from "next/image";
import Camera from "@/components/tools/camera";
import Header from "@/components/header";
import RecognitionDisplay from "@/components/recognition-display";
import { Camera as CameraIcon, User } from "lucide-react";
import { useToast } from "@/components/ui/use-toast";

export default function Home() {
  const [currentRecognizedPerson, setCurrentRecognizedPerson] = useState<Person | null>(null);
  const [showCameraSetup, setShowCameraSetup] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const { toast } = useToast();

  // Update the handler
  const handlePersonRecognized = useCallback((person: Person) => {
    console.log("Person recognized:", person);
    
    // Only update if we have a valid person
    if (person && person.name) {
      setCurrentRecognizedPerson({
        id: person.id,
        name: person.name,
        relationship: person.relationship || "Unknown",
        lastMet: person.lastMet || "Just now",
        description: person.description || "No additional information available.",
        imageUrl: "/placeholder-profile.jpg" // You could generate/store actual images
      });
      
      toast({
        title: "Person recognized",
        description: `Identified: ${person.name}`,
      });
    }
  }, [toast]);

  // Simulate recognition after camera has been active for a few seconds
  useEffect(() => {
    let timer: NodeJS.Timeout;
    
    if (isCameraActive) {
      // Only start recognition when camera is active
      timer = setTimeout(() => {
        setCurrentRecognizedPerson(mockPerson);
        toast({
          title: "Person recognized",
          description: "Identified: Sarah Johnson",
        });
      }, 5000);
    } else {
      // Clear recognized person when camera is off
      setCurrentRecognizedPerson(null);
    }
    
    return () => clearTimeout(timer);
  }, [isCameraActive, toast]);
  
  // Handle camera state changes
  const handleCameraStateChange = useCallback((isActive: boolean) => {
    // Update the camera active state
    setIsCameraActive(isActive);
    console.log(`Camera is ${isActive ? 'active' : 'inactive'}`);
    
    if (isActive) {
      toast({
        title: "Camera activated",
        description: "Looking for familiar faces...",
        duration: 3000,
      });
    }
  }, [toast]);

  // Handle camera button click
  const handleOpenCamera = useCallback(() => {
    setShowCameraSetup(true);
    toast({
      title: "Opening camera...",
      duration: 2000,
    });
  }, [toast]);

  return (
    <div className="min-h-screen bg-gray-50">
      <main className="pt-20 pb-10 px-4">
        {/* Main content */}
        <div className="max-w-4xl mx-auto mt-8">
          <div className="text-center mb-10">
            <h1 className="text-3xl font-bold text-gray-800">Memory Assistant</h1>
            <p className="text-gray-600 mt-2">
              Looking at someone? I'll help you remember who they are.
            </p>
          </div>
          
          {/* Recognition Display */}
          <RecognitionDisplay person={currentRecognizedPerson} />
          
          {/* Visual indicator when camera is active */}
          {isCameraActive && (
            <div className="fixed top-4 left-1/2 -translate-x-1/2 bg-blue-600/90 backdrop-blur-sm text-white px-3 py-1.5 rounded-full z-20 flex items-center shadow-lg">
              <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse mr-2"></div>
              <span className="text-sm font-medium">Camera Active</span>
            </div>
          )}
          
          {/* Camera control button - different styles for active/inactive */}
          <div className={`fixed ${isCameraActive ? 'bottom-6 left-6' : 'bottom-6 right-6'} z-30`}>
            <button
              onClick={isCameraActive ? () => {} : handleOpenCamera}
              className={`${
                isCameraActive 
                  ? "bg-gray-700/60 backdrop-blur-md pointer-events-none" 
                  : "bg-blue-600 hover:bg-blue-700"
              } text-white p-4 rounded-full shadow-lg transition-all duration-300`}
              aria-label={isCameraActive ? "Camera is active" : "Open camera"}
            >
              <CameraIcon size={24} className={isCameraActive ? "opacity-70" : ""} />
            </button>
          </div>
          
          
        </div>
        
        {/* Camera Component - pass showSetup prop to control visibility */}
        <Camera 
          onStateChange={handleCameraStateChange}
          initialShowSetup={showCameraSetup} 
          onSetupClose={() => setShowCameraSetup(false)}
          onPersonRecognized={handlePersonRecognized}
        />
      </main>
    </div>
  );
}

// Person type definition
type Person = {
  id: string;
  name: string;
  relationship: string;
  lastMet: string;
  description: string;
  imageUrl: string;
};
