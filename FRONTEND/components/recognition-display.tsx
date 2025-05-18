import { useEffect, useState } from 'react';
import Image from 'next/image';
import { User, Clock, Heart, CalendarCheck, MapPin, AlertCircle, BookOpen } from 'lucide-react';

type Person = {
  id: string;
  name: string;
  relationship: string;
  lastMet: string;
  description: string;
  imageUrl: string;
};

type RecognitionDisplayProps = {
  person: Person | null;
};

const RecognitionDisplay = ({ person }: RecognitionDisplayProps) => {
  const [isVisible, setIsVisible] = useState(false);
  const [animationState, setAnimationState] = useState('entering');
  
  useEffect(() => {
    if (person) {
      setAnimationState('entering');
      setIsVisible(true);
      
      // Add a slight delay before showing the full animation
      const timer = setTimeout(() => {
        setAnimationState('visible');
      }, 100);
      
      return () => clearTimeout(timer);
    } else {
      setAnimationState('exiting');
      const timer = setTimeout(() => {
        setIsVisible(false);
      }, 300);
      
      return () => clearTimeout(timer);
    }
  }, [person]);
  
  if (!isVisible) {
    return (
      <div className="flex flex-col items-center justify-center py-16 px-4 bg-white/80 backdrop-blur-sm rounded-xl shadow-sm border border-gray-100">
        <div className="w-24 h-24 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-full flex items-center justify-center mb-5 shadow-inner">
          <User size={40} className="text-blue-400" />
        </div>
        <h2 className="text-xl font-semibold text-gray-700">Waiting for Recognition</h2>
        <p className="text-gray-500 mt-2 text-center max-w-md">
          Look at someone through the camera and I'll help you remember who they are.
        </p>
        
        {/* Enhanced animation dots */}
        <div className="flex space-x-3 mt-8">
          {[0, 1, 2].map((i) => (
            <div 
              key={i}
              className="w-3 h-3 bg-gradient-to-r from-blue-400 to-blue-500 rounded-full shadow-sm"
              style={{ 
                animation: "pulse-scale 1.5s infinite ease-in-out",
                animationDelay: `${i * 0.2}s`
              }}
            ></div>
          ))}
        </div>
        
        {/* Add some helpful tips */}
        <div className="mt-10 bg-blue-50 px-6 py-4 rounded-lg border border-blue-100 max-w-md">
          <h3 className="font-medium text-blue-700 flex items-center">
            <AlertCircle size={16} className="mr-2" /> Helpful Tips
          </h3>
          <ul className="mt-2 text-sm text-blue-800/80 space-y-2">
            <li className="flex items-start">
              <span className="rounded-full bg-blue-100 p-1 mr-2 mt-0.5">
                <svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>
              </span>
              Make sure the person's face is clearly visible in the camera
            </li>
            <li className="flex items-start">
              <span className="rounded-full bg-blue-100 p-1 mr-2 mt-0.5">
                <svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>
              </span>
              Good lighting helps improve recognition accuracy
            </li>
          </ul>
        </div>
      </div>
    );
  }
  
  return (
    <div 
      className={`bg-white rounded-xl shadow-md overflow-hidden transition-all duration-500 transform
      ${animationState === 'entering' ? 'scale-95 opacity-0' : 
        animationState === 'exiting' ? 'scale-95 opacity-0' : 
        'scale-100 opacity-100'}`}
    >
      {/* Recognition banner */}
      <div className="bg-gradient-to-r from-green-50 to-emerald-50 px-4 py-3 flex items-center justify-center border-b border-green-100">
        <div className="w-2 h-2 bg-emerald-400 rounded-full mr-2 animate-pulse"></div>
        <span className="text-emerald-800 font-medium text-sm">Person Recognized</span>
      </div>
      
      <div className="p-6 md:p-8">
        <div className="flex flex-col md:flex-row items-center md:items-start gap-6">
          {/* Enhanced profile image with border effect */}
          <div className="w-36 h-36 rounded-full overflow-hidden border-4 border-white shadow-lg relative">
            <div className="absolute inset-0 bg-gradient-to-br from-blue-200/20 to-purple-200/20 z-10"></div>
            {person?.imageUrl ? (
              <Image 
                src={person.imageUrl} 
                alt={person.name} 
                width={144} 
                height={144}
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="w-full h-full bg-gradient-to-br from-blue-100 to-indigo-50 flex items-center justify-center">
                <User size={56} className="text-blue-400" />
              </div>
            )}
          </div>
          
          {/* Enhanced person details */}
          <div className="flex-1 text-center md:text-left">
            <h2 className="text-3xl font-bold text-gray-800 mb-2">{person?.name}</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-5">
              <div className="flex items-center text-gray-700 bg-gray-50 p-2 rounded-lg">
                <Heart size={16} className="mr-2 text-red-500" />
                <span className="font-medium">{person?.relationship}</span>
              </div>
              
              <div className="flex items-center text-gray-700 bg-gray-50 p-2 rounded-lg">
                <Clock size={16} className="mr-2 text-blue-500" />
                <span>Last seen: {person?.lastMet}</span>
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-blue-50 to-indigo-50 p-5 rounded-lg border border-blue-100 shadow-inner">
              <h3 className="flex items-center text-blue-800 font-medium mb-2">
                <BookOpen size={16} className="mr-2" />
                Memory Notes
              </h3>
              <p className="text-gray-700 leading-relaxed">
                {person?.description}
              </p>
            </div>
          </div>
        </div>
        
        {/* Additional information cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-8">
          <div className="bg-amber-50 rounded-lg p-4 border border-amber-100">
            <h3 className="flex items-center text-amber-800 font-medium mb-2">
              <CalendarCheck size={16} className="mr-2" />
              Upcoming Events
            </h3>
            <p className="text-amber-700 text-sm">
              Next scheduled visit: Tuesday at 3:00 PM
            </p>
          </div>
          
          <div className="bg-purple-50 rounded-lg p-4 border border-purple-100">
            <h3 className="flex items-center text-purple-800 font-medium mb-2">
              <MapPin size={16} className="mr-2" />
              Important Places
            </h3>
            <p className="text-purple-700 text-sm">
              Lives at: 1234 Maple Street, 20 minutes away
            </p>
          </div>
        </div>
        
        {/* Action buttons - now just the edit button */}
        <div className="mt-6 flex justify-center md:justify-end">
          <button className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors flex items-center shadow-sm">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path></svg>
            Edit Details
          </button>
        </div>
      </div>

      {/* Add a footer with confidence level */}
      <div className="bg-gray-50 px-6 py-3 border-t border-gray-100 flex items-center justify-between">
        <div className="flex items-center text-xs text-gray-500">
          <span>Recognition Confidence: </span>
          <div className="ml-2 bg-gray-200 h-2 w-24 rounded-full overflow-hidden">
            <div className="bg-green-500 h-full" style={{ width: "87%" }}></div>
          </div>
          <span className="ml-2">87%</span>
        </div>
        <div className="text-xs text-gray-500">
          Last updated: {new Date().toLocaleTimeString()}
        </div>
      </div>
    </div>
  );
};

export default RecognitionDisplay;

// Add this to your global CSS or use a style tag in the component
/*
@keyframes pulse-scale {
  0%, 100% { transform: scale(0.8); opacity: 0.5; }
  50% { transform: scale(1.2); opacity: 1; }
}
*/