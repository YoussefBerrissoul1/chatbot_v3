import { useState, useEffect, useCallback } from 'react';
import { debounce, throttle, getMemoryUsage, networkMonitor } from '../utils/performance';

export const usePerformance = () => {
  const [metrics, setMetrics] = useState({
    memory: null,
    networkStatus: navigator.onLine,
    connectionType: 'unknown',
    renderTime: 0
  });

  const updateMetrics = useCallback(() => {
    setMetrics(prev => ({
      ...prev,
      memory: getMemoryUsage(),
      networkStatus: networkMonitor.isOnline(),
      connectionType: networkMonitor.getConnectionType()
    }));
  }, []);

  const measureRenderTime = useCallback((componentName) => {
    const start = performance.now();
    return () => {
      const end = performance.now();
      const renderTime = end - start;
      setMetrics(prev => ({
        ...prev,
        renderTime
      }));
      console.log(`${componentName} render time: ${renderTime.toFixed(2)}ms`);
    };
  }, []);

  useEffect(() => {
    // Initial metrics
    updateMetrics();

    // Set up periodic monitoring
    const interval = setInterval(updateMetrics, 30000); // Every 30 seconds

    // Network status monitoring
    networkMonitor.onStatusChange((isOnline) => {
      setMetrics(prev => ({
        ...prev,
        networkStatus: isOnline
      }));
    });

    return () => {
      clearInterval(interval);
    };
  }, [updateMetrics]);

  return {
    metrics,
    measureRenderTime,
    updateMetrics
  };
};

export const useDebounce = (value, delay) => {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
};

export const useThrottle = (value, limit) => {
  const [throttledValue, setThrottledValue] = useState(value);

  useEffect(() => {
    const throttledFn = throttle(() => {
      setThrottledValue(value);
    }, limit);

    throttledFn();
  }, [value, limit]);

  return throttledValue;
};

export const useIntersectionObserver = (options = {}) => {
  const [isIntersecting, setIsIntersecting] = useState(false);
  const [ref, setRef] = useState(null);

  useEffect(() => {
    if (!ref) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        setIsIntersecting(entry.isIntersecting);
      },
      {
        threshold: 0.1,
        rootMargin: '50px',
        ...options
      }
    );

    observer.observe(ref);

    return () => {
      observer.disconnect();
    };
  }, [ref, options]);

  return [setRef, isIntersecting];
};

export const useLocalStorage = (key, initialValue) => {
  const [storedValue, setStoredValue] = useState(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.error(`Error reading localStorage key "${key}":`, error);
      return initialValue;
    }
  });

  const setValue = useCallback((value) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      window.localStorage.setItem(key, JSON.stringify(valueToStore));
    } catch (error) {
      console.error(`Error setting localStorage key "${key}":`, error);
    }
  }, [key, storedValue]);

  const removeValue = useCallback(() => {
    try {
      window.localStorage.removeItem(key);
      setStoredValue(initialValue);
    } catch (error) {
      console.error(`Error removing localStorage key "${key}":`, error);
    }
  }, [key, initialValue]);

  return [storedValue, setValue, removeValue];
};

export const useMediaQuery = (query) => {
  const [matches, setMatches] = useState(() => {
    if (typeof window !== 'undefined') {
      return window.matchMedia(query).matches;
    }
    return false;
  });

  useEffect(() => {
    if (typeof window === 'undefined') return;

    const mediaQuery = window.matchMedia(query);
    const handler = (event) => setMatches(event.matches);

    mediaQuery.addEventListener('change', handler);
    return () => mediaQuery.removeEventListener('change', handler);
  }, [query]);

  return matches;
};

export const useOnlineStatus = () => {
  const [isOnline, setIsOnline] = useState(navigator.onLine);

  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  return isOnline;
};

