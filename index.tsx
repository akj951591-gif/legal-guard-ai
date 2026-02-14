import React, { useState, useEffect, useRef, useMemo } from 'react';
import ReactDOM from 'react-dom/client';
import { GoogleGenAI, Type } from "@google/genai";
import { 
  ShieldAlert, Scale, Sun, Moon, Hash, Info, MapPin, History, 
  Send, Loader2, AlertCircle, FileQuestion, Gavel, 
  ShieldCheck, HeartHandshake, BookOpen, MessageCircleQuestion,
  Mic, MicOff, CheckCircle2, AlertTriangle, FileText, UserCheck, 
  Printer, Star, Landmark, Heart, Baby, Briefcase, Search, X,
  Plus, Trash2, Download, Save, Bold, Italic, List, Code, 
  Type as TypeIcon, ArrowLeft, Navigation, ExternalLink, Calendar,
  ChevronRight, Trash, Clock
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

// --- TYPES & INTERFACES ---

export enum CaseType {
  FALSE_CASE = "False Case/Allegation",
  LAND_DISPUTE = "Land & Property Dispute",
  MATRIMONIAL = "Matrimonial/Divorce",
  CRIMINAL = "Criminal/Violence",
  CYBER_CRIME = "Cyber Crime/Deepfake",
  CORRUPTION = "Corruption/Money Laundering",
  CHILD_ABUSE = "Child Abuse/POSCO",
  HARASSMENT = "Workplace/Women Harassment",
  DOMESTIC_VIOLENCE = "Domestic Violence/Dowry",
  STEALING = "Theft/Stealing/Robbery",
  OTHER = "Other Legal Issue"
}

export type View = 'home' | 'library' | 'qa' | 'sections' | 'about' | 'policeFinder' | 'history';

export interface CaseSubmission {
  type: CaseType;
  description: string;
  urgency: 'low' | 'medium' | 'high';
}

export interface LegalAnalysis {
  summary: string;
  immediateSteps: string[];
  legalProvisions: { section: string; description: string; }[];
  keyPersonnel: string[];
  documentChecklist: string[];
  policeRights: string[];
}

export interface StoredAnalysis {
  id: string;
  timestamp: number;
  submission: CaseSubmission;
  analysis: LegalAnalysis;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  text: string;
  timestamp: number;
}

export interface ChatSession {
  id: string;
  title: string;
  timestamp: number;
  messages: ChatMessage[];
}

// --- GEMINI SERVICES ---

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

const analyzeLegalProblem = async (submission: CaseSubmission): Promise<LegalAnalysis> => {
  const response = await ai.models.generateContent({
    model: "gemini-3-pro-preview",
    contents: `Analyze the following legal problem and provide a structured response. 
    User Case Type: ${submission.type}
    Description: ${submission.description}
    Urgency: ${submission.urgency}
    
    Response must follow the JSON schema. Include specific IPC/BNS sections.
    Explain critical rights: Art 22(1) (grounds), 24-hour production rule, Arrest Memo, Women's protection (no arrest after sunset).`,
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          summary: { type: Type.STRING },
          immediateSteps: { type: Type.ARRAY, items: { type: Type.STRING } },
          legalProvisions: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                section: { type: Type.STRING },
                description: { type: Type.STRING }
              },
              required: ["section", "description"]
            }
          },
          keyPersonnel: { type: Type.ARRAY, items: { type: Type.STRING } },
          documentChecklist: { type: Type.ARRAY, items: { type: Type.STRING } },
          policeRights: { type: Type.ARRAY, items: { type: Type.STRING } }
        },
        required: ["summary", "immediateSteps", "legalProvisions", "keyPersonnel", "documentChecklist", "policeRights"]
      }
    }
  });
  return JSON.parse(response.text || "{}") as LegalAnalysis;
};

const askLegalQuestion = async (question: string, history: { role: string, text: string }[]): Promise<string> => {
  const contents = history.map(h => ({
    role: h.role === 'user' ? 'user' : 'model',
    parts: [{ text: h.text }]
  }));
  contents.push({ role: 'user', parts: [{ text: question }] });
  const response = await ai.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: contents,
    config: {
      systemInstruction: "You are a specialized Indian legal assistant. Provide conversational, helpful information. Emphasize this isn't official advice. Use Markdown.",
    }
  });
  return response.text || "I'm sorry, I couldn't process that.";
};

const lookupLegalSection = async (query: string): Promise<string> => {
  const response = await ai.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: `Detailed explanation of legal section/term: "${query}". Include Act, simplified meaning, punishments, and notable exceptions. Use Markdown.`,
  });
  return response.text || "No information found.";
};

const findNearbyPoliceStations = async (latitude: number, longitude: number) => {
  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash',
    contents: "Find the 5 closest police stations to my current location. Provide name, address, and if available, contact info. Mention Zero FIR.",
    config: {
      tools: [{ googleMaps: {} }, { googleSearch: {} }],
      toolConfig: { retrievalConfig: { latLng: { latitude, longitude } } }
    },
  });
  return {
    text: response.text || "No stations found.",
    grounding: response.candidates?.[0]?.groundingMetadata?.groundingChunks || []
  };
};

// --- COMPONENTS ---

const Header: React.FC<{ currentView: View; onViewChange: (v: View) => void; theme: string; onToggleTheme: () => void; }> = ({ currentView, onViewChange, theme, onToggleTheme }) => {
  const isDarkMode = theme === 'dark';
  return (
    <header className={`shadow-xl sticky top-0 z-50 transition-colors duration-500 ${isDarkMode ? 'bg-slate-900 text-white' : 'bg-white text-slate-900'}`}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-20 flex items-center justify-between">
        <div className="flex items-center space-x-3 cursor-pointer" onClick={() => onViewChange('home')}>
          <div className="bg-amber-500 p-2 rounded-lg"><Scale size={28} className="text-slate-900" /></div>
          <div>
            <h1 className="text-2xl font-bold font-serif tracking-tight">LegalGuard AI</h1>
            <p className="text-xs text-amber-500 uppercase tracking-widest font-semibold">Instant Justice Protocol</p>
          </div>
        </div>
        <nav className="hidden lg:flex space-x-6 text-sm font-medium uppercase tracking-wider">
          {['home', 'qa', 'history', 'policeFinder', 'sections', 'about'].map((v) => (
            <button key={v} onClick={() => onViewChange(v as View)} className={`${currentView === v ? 'text-amber-400' : 'hover:text-amber-400'} transition flex items-center gap-1.5`}>
              {v === 'home' ? 'Analysis' : v === 'qa' ? 'Q&A' : v === 'policeFinder' ? <><MapPin size={14}/>Nearby</> : v.charAt(0).toUpperCase() + v.slice(1)}
            </button>
          ))}
        </nav>
        <div className="flex items-center space-x-4">
          <button onClick={onToggleTheme} className={`p-2 rounded-full ${isDarkMode ? 'bg-slate-800 text-amber-400' : 'bg-slate-100 text-slate-600'}`}>
            {isDarkMode ? <Sun size={20} /> : <Moon size={20} />}
          </button>
          <button className="bg-red-600 text-white px-4 py-2 rounded-full text-sm font-bold flex items-center gap-2 animate-pulse">
            <ShieldAlert size={18} /> SOS
          </button>
        </div>
      </div>
    </header>
  );
};

const LegalOutput: React.FC<{ analysis: LegalAnalysis; onReset: () => void; isDarkMode: boolean; }> = ({ analysis, onReset, isDarkMode }) => (
  <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
    <div className={`border-l-8 border-amber-500 rounded-xl shadow-lg p-6 md:p-8 ${isDarkMode ? 'bg-slate-900 text-white' : 'bg-white text-slate-900'}`}>
      <div className="flex justify-between items-start mb-6">
        <h2 className="text-3xl font-serif font-bold">Case Strategy & Analysis</h2>
        <button onClick={() => window.print()} className="flex items-center gap-2 text-slate-500 hover:text-slate-900 print:hidden">
          <Printer size={20} /> Print
        </button>
      </div>
      <p className="text-slate-400 italic mb-8 leading-relaxed text-lg">"{analysis.summary}"</p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <section className="bg-red-900/10 p-6 rounded-xl border border-red-500/20">
          <h3 className="flex items-center gap-2 text-red-500 font-bold mb-4 uppercase text-sm">Immediate Actions</h3>
          <ul className="space-y-3">
            {analysis.immediateSteps.map((step, i) => (
              <li key={i} className="flex gap-3 text-sm leading-snug">
                <span className="w-5 h-5 bg-red-500 text-white rounded-full flex items-center justify-center text-[10px] font-bold flex-shrink-0">{i+1}</span>
                {step}
              </li>
            ))}
          </ul>
        </section>
        <section className="bg-slate-800/50 p-6 rounded-xl border border-slate-700">
          <h3 className="flex items-center gap-2 text-amber-500 font-bold mb-4 uppercase text-sm">Legal Provisions</h3>
          <div className="space-y-4">
            {analysis.legalProvisions.map((p, i) => (
              <div key={i} className="border-b border-slate-700 pb-3 last:border-0">
                <div className="text-amber-400 font-bold text-xs mb-1">{p.section}</div>
                <div className="text-slate-400 text-xs">{p.description}</div>
              </div>
            ))}
          </div>
        </section>
      </div>
      <div className="mt-8 flex justify-center pt-8 border-t border-slate-800">
        <button onClick={onReset} className="bg-amber-500 text-slate-900 px-8 py-3 rounded-lg font-bold hover:bg-amber-600 transition print:hidden">Start New Case</button>
      </div>
    </div>
  </div>
);

const SupportModal: React.FC<{ isOpen: boolean; onClose: () => void; isDarkMode: boolean; }> = ({ isOpen, onClose, isDarkMode }) => {
  if (!isOpen) return null;
  const contacts = [
    { name: "Emergency", phone: "112", color: "text-red-500" },
    { name: "Women Helpline", phone: "1091", color: "text-pink-500" },
    { name: "Child Help", phone: "1098", color: "text-blue-500" }
  ];
  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm" onClick={onClose}>
      <div className={`w-full max-w-md rounded-2xl p-6 ${isDarkMode ? 'bg-slate-900 border border-slate-800' : 'bg-white'}`} onClick={e => e.stopPropagation()}>
        <div className="flex justify-between items-center mb-6">
          <h3 className="text-xl font-bold font-serif text-amber-500">Legal Aid Support</h3>
          <button onClick={onClose}><X /></button>
        </div>
        <div className="space-y-4">
          {contacts.map((c, i) => (
            <div key={i} className="flex justify-between p-3 border border-slate-800 rounded-lg">
              <span className={`font-bold ${c.color}`}>{c.name}</span>
              <a href={`tel:${c.phone}`} className="font-mono text-amber-500">{c.phone}</a>
            </div>
          ))}
        </div>
        <button onClick={onClose} className="w-full mt-6 py-3 bg-slate-800 text-white rounded-lg font-bold">Close</button>
      </div>
    </div>
  );
};

// --- MAIN APP ---

const App: React.FC = () => {
  const [view, setView] = useState<View>('home');
  const [theme, setTheme] = useState<'light' | 'dark'>(() => (localStorage.getItem('theme') as any) || 'dark');
  const [loading, setLoading] = useState(false);
  const [analysis, setAnalysis] = useState<LegalAnalysis | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submission, setSubmission] = useState<CaseSubmission>({ type: CaseType.FALSE_CASE, description: '', urgency: 'medium' });
  const [isSupportOpen, setIsSupportOpen] = useState(false);

  useEffect(() => {
    localStorage.setItem('theme', theme);
    document.documentElement.classList.toggle('dark', theme === 'dark');
  }, [theme]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!submission.description.trim()) return setError("Please describe your situation.");
    setLoading(true);
    setError(null);
    try {
      const result = await analyzeLegalProblem(submission);
      setAnalysis(result);
    } catch (err) {
      setError("Analysis failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const renderHome = () => (
    analysis ? <LegalOutput analysis={analysis} onReset={() => setAnalysis(null)} isDarkMode={theme === 'dark'} /> : (
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-10">
          <h2 className="text-4xl font-serif font-bold mb-4">Immediate Legal Assistance</h2>
          <p className="text-slate-400">Describe your problem. AI provides instant rights, laws, and evidence steps.</p>
        </div>
        <div className={`p-8 rounded-2xl border ${theme === 'dark' ? 'bg-slate-900/50 border-slate-800' : 'bg-white'}`}>
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-xs font-bold uppercase mb-2">Category</label>
                <select 
                  value={submission.type} 
                  onChange={e => setSubmission({...submission, type: e.target.value as CaseType})}
                  className="w-full p-3 rounded bg-slate-800 border border-slate-700 text-white"
                >
                  {Object.values(CaseType).map(t => <option key={t} value={t}>{t}</option>)}
                </select>
              </div>
              <div>
                <label className="block text-xs font-bold uppercase mb-2">Urgency</label>
                <div className="flex gap-2 bg-slate-800 p-1 rounded">
                  {['low', 'medium', 'high'].map(u => (
                    <button key={u} type="button" onClick={() => setSubmission({...submission, urgency: u as any})} className={`flex-1 py-1 text-xs font-bold rounded ${submission.urgency === u ? 'bg-amber-500 text-slate-900' : 'text-slate-400'}`}>
                      {u.toUpperCase()}
                    </button>
                  ))}
                </div>
              </div>
            </div>
            <textarea 
              rows={5} 
              value={submission.description} 
              onChange={e => setSubmission({...submission, description: e.target.value})}
              placeholder="Explain the incident in detail..."
              className="w-full p-4 rounded bg-slate-800 border border-slate-700 text-white"
            />
            {error && <p className="text-red-500 text-sm font-bold flex items-center gap-1"><AlertCircle size={14}/>{error}</p>}
            <button disabled={loading} className="w-full py-4 bg-amber-500 text-slate-900 rounded-xl font-bold flex items-center justify-center gap-2 hover:bg-amber-600 transition disabled:opacity-50">
              {loading ? <><Loader2 className="animate-spin" /> Processing...</> : <><Send size={20}/> Analyze Case</>}
            </button>
          </form>
        </div>
      </div>
    )
  );

  return (
    <div className={`min-h-screen flex flex-col ${theme === 'dark' ? 'bg-slate-950 text-slate-200' : 'bg-slate-50 text-slate-900'}`}>
      <Header currentView={view} onViewChange={setView} theme={theme} onToggleTheme={() => setTheme(theme === 'light' ? 'dark' : 'light')} />
      <main className="flex-grow container mx-auto px-4 py-12">
        {view === 'home' ? renderHome() : <div className="text-center py-20 text-slate-500">Module {view} simplified. Check dashboard for main analysis.</div>}
      </main>
      <button onClick={() => setIsSupportOpen(true)} className="fixed bottom-8 right-8 bg-amber-500 text-slate-900 p-4 rounded-full shadow-2xl font-bold flex items-center gap-2 transition hover:scale-105">
        <HeartHandshake /><span className="hidden md:inline">Legal Aid</span>
      </button>
      <SupportModal isOpen={isSupportOpen} onClose={() => setIsSupportOpen(false)} isDarkMode={theme === 'dark'} />
    </div>
  );
};

const root = ReactDOM.createRoot(document.getElementById('root')!);
root.render(<App />);