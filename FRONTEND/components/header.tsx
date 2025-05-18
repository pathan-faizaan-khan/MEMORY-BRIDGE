'use client';
import React, { useState, useEffect } from 'react';
import { 
  Users, 
  PlusCircle, 
  ListTodo, 
  CalendarCheck, 
  Menu, 
  X, 
  Brain, 
  Bell,
  User
} from 'lucide-react';
import Link from 'next/link';

const Header = () => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [activeLink, setActiveLink] = useState('home');

  // Handle scroll effect for header
  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 10);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const navigation = [
    { id: 'people', label: 'Saved People', icon: <Users className="w-5 h-5" /> },
    { id: 'add-person', label: 'Add Person', icon: <PlusCircle className="w-5 h-5" /> },
    { id: 'tasks', label: 'Tasks', icon: <ListTodo className="w-5 h-5" /> },
    { id: 'routine', label: 'Daily Routine', icon: <CalendarCheck className="w-5 h-5" /> }
  ];

  return (
    <header 
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        isScrolled 
          ? 'bg-white shadow-md py-2' 
          : 'bg-white/80 backdrop-blur-md py-4'
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center">
          {/* Logo */}
          <Link href="/" className="flex items-center space-x-3" onClick={() => setActiveLink('home')}>
            <div className="bg-gradient-to-r from-blue-600 to-indigo-700 p-2 rounded-lg shadow-md">
              <Brain className="h-6 w-6 text-white" />
            </div>
            <span className="font-bold text-xl text-gray-800">Memory Assist</span>
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center space-x-1">
            {navigation.map((item) => (
              <Link
                key={item.id}
                href={`/${item.id}`}
                className={`flex items-center px-3 py-2 rounded-full text-sm font-medium transition-colors ${
                  activeLink === item.id 
                    ? 'bg-blue-50 text-blue-700' 
                    : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                }`}
                onClick={() => setActiveLink(item.id)}
              >
                <span className="flex items-center">
                  {item.icon}
                  <span className="ml-2">{item.label}</span>
                </span>
              </Link>
            ))}
          </nav>

          {/* Right side buttons */}
          <div className="hidden md:flex items-center space-x-3">
            <button className="p-2 rounded-full text-gray-500 hover:bg-gray-100 relative">
              <Bell className="h-5 w-5" />
              <span className="absolute top-0 right-0 h-2 w-2 bg-red-500 rounded-full"></span>
            </button>
            <button className="flex items-center space-x-2 p-1.5 rounded-full border border-gray-200 hover:bg-gray-50">
              <div className="bg-blue-100 text-blue-800 rounded-full p-1">
                <User className="h-4 w-4" />
              </div>
              <span className="text-sm font-medium text-gray-700 pr-1">Profile</span>
            </button>
          </div>

          {/* Mobile menu button */}
          <button 
            className="md:hidden p-2 rounded-md text-gray-500 hover:bg-gray-100"
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
          >
            {isMobileMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      {isMobileMenuOpen && (
        <div className="md:hidden">
          <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3 border-t border-gray-200 bg-white shadow-lg">
            {navigation.map((item) => (
              <Link
                key={item.id}
                href={`/${item.id}`}
                className={`flex items-center px-3 py-2 rounded-md text-base font-medium ${
                  activeLink === item.id 
                    ? 'bg-blue-50 text-blue-700' 
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
                onClick={() => {
                  setActiveLink(item.id);
                  setIsMobileMenuOpen(false);
                }}
              >
                {item.icon}
                <span className="ml-3">{item.label}</span>
              </Link>
            ))}
            <div className="pt-4 pb-1 border-t border-gray-200">
              <div className="flex items-center px-4">
                <div className="flex-shrink-0">
                  <div className="bg-blue-100 text-blue-800 rounded-full p-2">
                    <User className="h-5 w-5" />
                  </div>
                </div>
                <div className="ml-3">
                  <div className="text-base font-medium text-gray-800">User Profile</div>
                  <div className="text-sm font-medium text-gray-500">View settings</div>
                </div>
                <button className="ml-auto p-2 rounded-full text-gray-500 hover:bg-gray-100">
                  <Bell className="h-5 w-5" />
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </header>
  );
};

export default Header;
