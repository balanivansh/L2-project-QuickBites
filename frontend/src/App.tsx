import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { MapContainer, TileLayer, Marker, Popup, Polyline, Tooltip } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { MapPin, Clock, Truck, AlertCircle, Sparkles, Navigation, Coffee, Map } from 'lucide-react';
import L from 'leaflet';

// Fix leaflet default icon issue
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';
let DefaultIcon = L.icon({
    iconUrl: icon,
    shadowUrl: iconShadow,
    iconSize: [25, 41],
    iconAnchor: [12, 41]
});
L.Marker.prototype.options.icon = DefaultIcon;

const restaurantIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-orange.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41]
});

const customerIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-blue.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41]
});

type Options = {
  restaurants: string[];
  customers: string[];
  days: string[];
};

type PredictionResult = {
  predicted_time: number;
  prep_time: number;
  travel_time: number;
  dist_km: number;
  estimated_fee: number;
  start_coords: [number, number];
  end_coords: [number, number];
  route_points: [number, number][];
};

function App() {
  const [options, setOptions] = useState<Options | null>(null);
  const [loadingOptions, setLoadingOptions] = useState(true);
  
  const [selectedRestaurant, setSelectedRestaurant] = useState('');
  const [selectedCustomer, setSelectedCustomer] = useState('');
  const [prepTime, setPrepTime] = useState(20);
  const [orderHour, setOrderHour] = useState(17);
  const [selectedDay, setSelectedDay] = useState('Monday');

  const [loadingPrediction, setLoadingPrediction] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

  useEffect(() => {
    axios.get(`${API_BASE_URL}/api/options`)
      .then(res => {
        setOptions(res.data);
        if (res.data.restaurants.length > 0) {
          const bk = res.data.restaurants.find((r: string) => r.includes('Burger King, Koramangala'));
          setSelectedRestaurant(bk || res.data.restaurants[0]);
        }
        if (res.data.customers.length > 0) {
          const wf = res.data.customers.find((c: string) => c.includes('Whitefield'));
          setSelectedCustomer(wf || res.data.customers[0]);
        }
        setLoadingOptions(false);
      })
      .catch(err => {
        console.error(err);
        setError('Connection failed. Make sure the backend AI engine is online.');
        setLoadingOptions(false);
      });
  }, []);

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoadingPrediction(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/predict`, {
        restaurant_name: selectedRestaurant,
        customer_location: selectedCustomer,
        prep_time: prepTime,
        order_hour: orderHour,
        day_name: selectedDay
      });
      setResult(response.data);
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.detail || 'Our ML engine encountered an anomaly.');
    } finally {
      setTimeout(() => setLoadingPrediction(false), 600); // minimal delay for animation smoothness
    }
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 font-sans relative overflow-hidden">
      {/* Dynamic Background Glowing Orbs */}
      <div className="absolute top-[-10%] left-[-10%] w-[50%] h-[50%] rounded-full bg-orange-600/20 blur-[150px] pointer-events-none" />
      <div className="absolute bottom-[-20%] right-[-10%] w-[60%] h-[60%] rounded-full bg-indigo-600/10 blur-[180px] pointer-events-none" />

      {/* Navbar Structure */}
      <nav className="fixed top-0 inset-x-0 h-20 glass z-50 flex items-center px-8 border-b border-zinc-800/50">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-orange-400 to-orange-600 flex items-center justify-center shadow-lg shadow-orange-500/20 animate-float">
            <Truck className="text-zinc-950 w-6 h-6" />
          </div>
          <span className="text-2xl font-black tracking-tight text-white">
            Quick<span className="text-orange-500">Bites</span>
          </span>
        </div>
        <div className="ml-auto flex items-center gap-4 text-sm font-medium text-zinc-400">
          <span className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" /> AI Engine Online</span>
        </div>
      </nav>

      {/* Main Content */}
      <main className="pt-32 pb-20 px-4 sm:px-8 max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8 relative z-10">
        
        {/* Left Column: Form */}
        <div className="col-span-1 lg:col-span-5 flex flex-col space-y-8 animate-slide-up">
          <div className="space-y-4">
            <h1 className="text-5xl lg:text-6xl font-black leading-tight tracking-tight">
              Predict <br/>
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-orange-400 to-amber-400">
                With Precision.
              </span>
            </h1>
            <p className="text-zinc-400 text-lg leading-relaxed max-w-md">
              Harness the power of real-time traffic data and our advanced machine learning model to perfectly time your logistics in Bengaluru.
            </p>
          </div>

          <div className="glass-card p-8 relative overflow-hidden group transition-all duration-500 hover:border-orange-500/30">
            {/* Subtle card glow */}
            <div className="absolute inset-0 bg-gradient-to-br from-orange-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none" />
            
            {error && (
              <div className="bg-red-500/10 border border-red-500/20 p-4 rounded-xl mb-6 flex items-start text-red-400 text-sm">
                <AlertCircle className="w-5 h-5 mr-3 flex-shrink-0 mt-0.5" />
                <p>{error}</p>
              </div>
            )}

            {loadingOptions ? (
              <div className="space-y-6 animate-pulse">
                {[1,2,3].map(i => (
                  <div key={i}>
                    <div className="h-4 w-24 bg-zinc-800 rounded mb-3" />
                    <div className="h-12 w-full bg-zinc-800/50 rounded-xl" />
                  </div>
                ))}
              </div>
            ) : (
              <form onSubmit={handlePredict} className="space-y-6 relative z-10">
                <div className="space-y-5">
                  <div className="space-y-2">
                    <label className="text-xs font-bold text-zinc-400 uppercase tracking-wider flex items-center gap-2">
                      <Coffee className="w-4 h-4 text-orange-500" /> Origin Restaurant
                    </label>
                    <div className="relative">
                      <select 
                        value={selectedRestaurant} onChange={e => setSelectedRestaurant(e.target.value)}
                        className="w-full appearance-none bg-zinc-900 border border-zinc-700 rounded-xl px-4 py-3.5 text-zinc-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent transition-all shadow-inner"
                      >
                        {options?.restaurants.map(r => <option key={r} value={r}>{r}</option>)}
                      </select>
                      <MapPin className="absolute right-4 top-1/2 -translate-y-1/2 w-5 h-5 text-zinc-500 pointer-events-none" />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <label className="text-xs font-bold text-zinc-400 uppercase tracking-wider flex items-center gap-2">
                      <Navigation className="w-4 h-4 text-blue-500" /> Destination
                    </label>
                    <div className="relative">
                      <select 
                        value={selectedCustomer} onChange={e => setSelectedCustomer(e.target.value)}
                        className="w-full appearance-none bg-zinc-900 border border-zinc-700 rounded-xl px-4 py-3.5 text-zinc-100 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all shadow-inner"
                      >
                        {options?.customers.map(c => <option key={c} value={c}>{c}</option>)}
                      </select>
                      <MapPin className="absolute right-4 top-1/2 -translate-y-1/2 w-5 h-5 text-zinc-500 pointer-events-none" />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <label className="text-xs font-bold text-zinc-400 uppercase tracking-wider">Day of Week</label>
                    <select 
                      value={selectedDay} onChange={e => setSelectedDay(e.target.value)}
                      className="w-full appearance-none bg-zinc-900 border border-zinc-700 rounded-xl px-4 py-3.5 text-zinc-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent transition-all shadow-inner"
                    >
                      {options?.days.map(d => <option key={d} value={d}>{d}</option>)}
                    </select>
                  </div>
                </div>

                <div className="pt-4 border-t border-zinc-800/80 space-y-8">
                  <div className="relative">
                    <div className="flex justify-between items-end mb-4">
                      <label className="text-sm font-semibold text-zinc-300">Preparation Time</label>
                      <span className="px-3 py-1 bg-zinc-800 text-orange-400 font-bold rounded-lg text-sm border border-zinc-700">{prepTime} min</span>
                    </div>
                    <input 
                      type="range" min="5" max="60" step="5" 
                      value={prepTime} onChange={e => setPrepTime(Number(e.target.value))}
                      className="w-full"
                    />
                  </div>
                  
                  <div className="relative">
                    <div className="flex justify-between items-end mb-4">
                      <label className="text-sm font-semibold text-zinc-300">Dispatch Hour</label>
                      <span className="px-3 py-1 bg-zinc-800 text-orange-400 font-bold rounded-lg text-sm border border-zinc-700">{orderHour}:00</span>
                    </div>
                    <input 
                      type="range" min="0" max="23" step="1" 
                      value={orderHour} onChange={e => setOrderHour(Number(e.target.value))}
                      className="w-full"
                    />
                  </div>
                </div>

                <button 
                  type="submit" 
                  disabled={loadingPrediction}
                  className={`w-full group mt-4 relative flex items-center justify-center gap-2 py-4 px-6 rounded-xl font-bold text-lg overflow-hidden transition-all duration-300
                    ${loadingPrediction 
                      ? 'bg-zinc-800 text-zinc-500 cursor-not-allowed border border-zinc-700' 
                      : 'bg-gradient-to-r from-orange-500 to-amber-500 text-zinc-950 hover:shadow-lg neon-glow hover:scale-[1.02] active:scale-95'
                    }
                  `}
                >
                  {loadingPrediction ? (
                    <>
                      <div className="w-5 h-5 border-2 border-zinc-600 border-t-zinc-400 rounded-full animate-spin" />
                      Computing Logistics...
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-5 h-5 transition-transform group-hover:rotate-12" />
                      Calculate ETA Forecast
                    </>
                  )}
                  {!loadingPrediction && (
                    <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300 ease-out pointer-events-none" />
                  )}
                </button>
              </form>
            )}
          </div>
        </div>

        {/* Right Column: Results & Map */}
        <div className="col-span-1 lg:col-span-7 flex flex-col space-y-6">
          {!result && !loadingPrediction && (
            <div className="h-full w-full min-h-[500px] glass-card flex flex-col items-center justify-center text-zinc-500 animate-slide-up" style={{animationDelay: '0.1s'}}>
              <Map className="w-16 h-16 mb-4 text-zinc-700" />
              <p className="text-lg font-medium">Ready for deployment.</p>
              <p className="text-sm">Input parameters to visualize analytics.</p>
            </div>
          )}

          {loadingPrediction && (
             <div className="h-full w-full min-h-[500px] glass-card flex flex-col items-center justify-center animate-pulse">
                <div className="w-16 h-16 border-4 border-zinc-800 border-t-orange-500 rounded-full animate-spin mb-6" />
                <div className="h-6 w-48 bg-zinc-800 rounded font-medium" />
             </div>
          )}

          {result && !loadingPrediction && (
            <>
              {/* Metrics Grid */}
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 animate-slide-up" style={{animationDelay: '0.1s'}}>
                <div className="glass-card p-5 relative overflow-hidden flex flex-col justify-center">
                  <div className="absolute -right-4 -top-4 w-16 h-16 bg-orange-500/20 blur-2xl rounded-full" />
                  <span className="text-zinc-500 text-xs font-bold uppercase tracking-widest mb-2 flex items-center gap-1.5"><Clock className="w-3.5 h-3.5"/> Total ETA</span>
                  <div className="flex items-baseline gap-1">
                    <span className="text-4xl font-black text-white">{Math.round(result.predicted_time)}</span>
                    <span className="text-orange-400 font-medium tracking-tight">min</span>
                  </div>
                </div>
                
                <div className="glass-card p-5 flex flex-col justify-center">
                  <span className="text-zinc-500 text-xs font-bold uppercase tracking-widest mb-2">Food Prep</span>
                  <div className="flex items-baseline gap-1">
                    <span className="text-3xl font-bold text-zinc-200">{Math.round(result.prep_time)}</span>
                    <span className="text-zinc-500 font-medium text-sm">min</span>
                  </div>
                </div>

                <div className="glass-card p-5 flex flex-col justify-center">
                  <span className="text-zinc-500 text-xs font-bold uppercase tracking-widest mb-2">Travel</span>
                  <div className="flex items-baseline gap-1">
                    <span className="text-3xl font-bold text-zinc-200">{Math.round(result.travel_time)}</span>
                    <span className="text-zinc-500 font-medium text-sm">min ({result.dist_km.toFixed(1)}km)</span>
                  </div>
                </div>

                <div className="glass-card p-5 relative overflow-hidden flex flex-col justify-center">
                   <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/10 to-transparent pointer-events-none" />
                  <span className="text-emerald-500/80 text-xs font-bold uppercase tracking-widest mb-2">Surge Fee</span>
                  <div className="flex items-baseline gap-1">
                    <span className="text-emerald-400 font-bold text-2xl">₹</span>
                    <span className="text-3xl font-black text-white">{result.estimated_fee.toFixed(0)}</span>
                  </div>
                </div>
              </div>

              {/* Map View */}
              <div className="h-[500px] lg:h-full min-h-[400px] glass-card p-2 animate-slide-up" style={{animationDelay: '0.2s'}}>
                <div className="w-full h-full rounded-2xl overflow-hidden relative">
                  <div className="absolute inset-0 ring-1 ring-inset ring-white/10 z-20 pointer-events-none rounded-2xl" />
                  
                  {result.route_points.length > 0 && (
                    <MapContainer 
                        center={result.route_points[Math.floor(result.route_points.length / 2)] as [number, number]} 
                        zoom={13} 
                        style={{ height: '100%', width: '100%', background: '#09090b' }}
                        zoomControl={false}
                    >
                      {/* Dark themed map tiles */}
                      <TileLayer
                        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>'
                        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                      />
                      <Marker position={result.start_coords} icon={restaurantIcon}>
                        <Tooltip permanent direction="top" className="font-bold text-indigo-900">
                          {selectedRestaurant}
                        </Tooltip>
                        <Popup className="custom-popup">
                          <div className="font-bold text-zinc-900">{selectedRestaurant}</div>
                          <div className="text-orange-600 text-xs">Origin Point</div>
                        </Popup>
                      </Marker>
                      <Marker position={result.end_coords} icon={customerIcon}>
                        <Tooltip permanent direction="top" className="font-bold text-indigo-900">
                          {selectedCustomer}
                        </Tooltip>
                        <Popup>
                          <div className="font-bold text-zinc-900">{selectedCustomer}</div>
                          <div className="text-blue-600 text-xs">Destination</div>
                        </Popup>
                      </Marker>
                      
                      {/* Beautiful glowing route line */}
                      <Polyline 
                        positions={result.route_points} 
                        color="#f97316" 
                        weight={5} 
                        opacity={0.8}
                        className="animate-pulse"
                      />
                      <Polyline 
                        positions={result.route_points} 
                        color="#f97316" 
                        weight={12} 
                        opacity={0.2}
                      />
                    </MapContainer>
                  )}
                </div>
              </div>
            </>
          )}

        </div>
      </main>
    </div>
  );
}

export default App;
