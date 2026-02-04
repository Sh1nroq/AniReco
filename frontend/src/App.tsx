import { useState } from 'react';
import { Check, ChevronsUpDown, Search, Star, Calendar } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList,
} from "@/components/ui/command";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";

// --- 1. ИНТЕРФЕЙСЫ (ТИПЫ) ---

interface Anime {
  mal_id: number;
  title: string;
  description: string;
  score?: number;
  image_url?: string;
  status?: string;
}

interface FilterOption {
  value: string;
  label: string;
}

interface RecommendationResponse {
  model_response: Anime[];
}

interface MultiSelectProps {
  selected: string[];
  setSelected: (genres: string[]) => void;
}

interface SimpleSelectProps {
  placeholder: string;
  options: FilterOption[];
  value: string;
  setValue: (v: string) => void;
}

// --- 2. КОНСТАНТЫ ---

const GENRES = [
  "Action", "Adventure", "Comedy", "Drama", "Sci-Fi", "Slice of Life",
  "Fantasy", "Romance", "Mystery", "Horror", "Sports", "Supernatural", "Suspense"
];

const TYPES: FilterOption[] = [
  { value: "TV", label: "TV Series" },
  { value: "Movie", label: "Movie" },
  { value: "OVA", label: "OVA" },
];

// --- 3. ГЛАВНЫЙ КОМПОНЕНТ ---

export default function AnimeApp() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<Anime[]>([]);

  // Состояния фильтров
  const [selectedGenres, setSelectedGenres] = useState<string[]>([]);
  const [yearMin, setYearMin] = useState<string>("");
  const [yearMax, setYearMax] = useState<string>("");
  const [minScore, setMinScore] = useState<string>("");
  const [type, setType] = useState("");

  const handleSearch = async () => {
    if (!query.trim() && selectedGenres.length === 0 && !type && !yearMin && !yearMax && !minScore) return;

    setLoading(true);

    try {
      const response = await fetch('http://127.0.0.1:8000/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text_query: query,
          genres: selectedGenres.length > 0 ? selectedGenres : null,
          type: type || null,
          year_min: yearMin ? parseInt(yearMin) : null,
          year_max: yearMax ? parseInt(yearMax) : null,
          min_score: minScore ? parseFloat(minScore) : null,
        }),
      });

      if (!response.ok) {
        console.error("API Error");
        setLoading(false);
        return;
      }

      const data: RecommendationResponse = await response.json();
      setResults(data.model_response || []);
    } catch (error) {
      console.error("Network Error:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-50 font-sans selection:bg-zinc-800">
      <main className="max-w-6xl mx-auto px-4 py-16 space-y-10">

        {/* HEADER */}
        <div className="space-y-4 text-center">
          <h1 className="text-5xl md:text-7xl font-black tracking-tighter bg-gradient-to-b from-white to-zinc-500 bg-clip-text text-transparent uppercase">
            AniReco
          </h1>
          <p className="text-zinc-500 text-lg font-medium">Neural Semantic Search Engine</p>
        </div>

        {/* SEARCH PANEL */}
        <section className="bg-zinc-900/40 border border-zinc-800/50 p-6 md:p-8 rounded-3xl backdrop-blur-xl space-y-8 shadow-2xl">

          <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
            {/* Multi-Genre Select */}
            <div className="md:col-span-5 space-y-2">
              <label className="text-[10px] font-bold uppercase tracking-widest text-zinc-500 ml-1">Genres</label>
              <MultiSelectGenres selected={selectedGenres} setSelected={setSelectedGenres} />
            </div>

            {/* Year Interval */}
            <div className="md:col-span-4 space-y-2">
              <label className="text-[10px] font-bold uppercase tracking-widest text-zinc-500 ml-1">Release Period</label>
              <div className="flex items-center gap-2">
                <div className="relative flex-1">
                  <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-zinc-600" />
                  <Input
                    type="number"
                    placeholder="From"
                    value={yearMin}
                    onChange={(e) => setYearMin(e.target.value)}
                    className="bg-zinc-950 border-zinc-800 pl-9 h-11 rounded-xl focus-visible:ring-zinc-700 [appearance:textfield]"
                  />
                </div>
                <span className="text-zinc-700">—</span>
                <div className="relative flex-1">
                  <Input
                    type="number"
                    placeholder="To"
                    value={yearMax}
                    onChange={(e) => setYearMax(e.target.value)}
                    className="bg-zinc-950 border-zinc-800 h-11 rounded-xl focus-visible:ring-zinc-700 [appearance:textfield]"
                  />
                </div>
              </div>
            </div>

            {/* Type Select */}
            <div className="md:col-span-3 space-y-2">
              <label className="text-[10px] font-bold uppercase tracking-widest text-zinc-500 ml-1">Format</label>
              <SimpleSelect
                placeholder="All Formats"
                options={TYPES}
                value={type}
                setValue={setType}
              />
            </div>
          </div>

          {/* MAIN SEARCH BAR */}
          <div className="flex flex-col md:flex-row gap-3">
            <div className="relative flex-1 group">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-zinc-500 group-focus-within:text-zinc-200 transition-colors" />
              <Input
                placeholder="Search by vibe: 'I want a dark cyberpunk with an invincible protagonist'..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                className="pl-12 pr-28 h-16 bg-zinc-950 border-zinc-800 text-lg rounded-2xl focus-visible:ring-zinc-700 placeholder:text-zinc-600"
              />

              {/* RATING INPUT inside Main Bar */}
              <div className="absolute right-4 top-1/2 -translate-y-1/2 flex items-center gap-2 border-l border-zinc-800 pl-4">
                <Star className="h-4 w-4 text-yellow-500 fill-yellow-500/20" />
                <input
                  type="number"
                  step="0.1"
                  min="0"
                  max="10"
                  placeholder="0.0"
                  value={minScore}
                  onChange={(e) => setMinScore(e.target.value)}
                  className="w-12 bg-transparent border-none text-sm font-bold focus:outline-none text-yellow-500 placeholder:text-zinc-800 [appearance:textfield]"
                />
              </div>
            </div>

            <Button
              onClick={handleSearch}
              disabled={loading}
              className="h-16 px-10 bg-zinc-50 text-zinc-950 hover:bg-white font-black rounded-2xl transition-all active:scale-95 shadow-xl"
            >
              {loading ? "SEARCHING..." : "FIND ANIME"}
            </Button>
          </div>
        </section>

        {/* RESULTS GRID */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {results.map((anime) => (
            <AnimeCard key={anime.mal_id} anime={anime} />
          ))}
        </div>

        {!loading && results.length === 0 && (
          <div className="text-center py-20 border-2 border-dashed border-zinc-900 rounded-3xl">
            <p className="text-zinc-600 font-medium italic">Your next favorite story is just one search away.</p>
          </div>
        )}
      </main>
    </div>
  );
}

// --- UI COMPONENTS ---

function MultiSelectGenres({ selected, setSelected }: MultiSelectProps) {
  const [open, setOpen] = useState(false);

  const toggleGenre = (genre: string) => {
    setSelected(
      selected.includes(genre)
        ? selected.filter(g => g !== genre)
        : [...selected, genre]
    );
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button variant="outline" className="w-full justify-between h-11 bg-zinc-950 border-zinc-800 rounded-xl text-zinc-400 font-normal">
          <span className="truncate">
            {selected.length > 0 ? `Selected: ${selected.length}` : "All Genres"}
          </span>
          <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[300px] p-0 bg-zinc-950 border-zinc-800 shadow-2xl">
        <Command className="bg-zinc-950">
          <CommandInput placeholder="Search genre..." className="text-zinc-200" />
          <CommandList>
            <CommandEmpty>No results.</CommandEmpty>
            <CommandGroup className="max-h-64 overflow-y-auto">
              {GENRES.map((g) => (
                <CommandItem key={g} onSelect={() => toggleGenre(g)} className="cursor-pointer text-zinc-400 aria-selected:bg-zinc-800 aria-selected:text-zinc-50">
                  <Check className={cn("mr-2 h-4 w-4 text-white", selected.includes(g) ? "opacity-100" : "opacity-0")} />
                  {g}
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}

function SimpleSelect({ placeholder, options, value, setValue }: SimpleSelectProps) {
  const [open, setOpen] = useState(false);
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button variant="outline" className="w-full justify-between h-11 bg-zinc-950 border-zinc-800 rounded-xl text-zinc-400 font-normal">
          {value ? options.find((o) => o.value === value)?.label : placeholder}
          <ChevronsUpDown className="h-4 w-4 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[200px] p-0 bg-zinc-950 border-zinc-800 shadow-2xl">
        <Command className="bg-zinc-950">
          <CommandList>
            {options.map((opt) => (
              <CommandItem
                key={opt.value}
                onSelect={() => { setValue(opt.value === value ? "" : opt.value); setOpen(false); }}
                className="cursor-pointer text-zinc-400 aria-selected:bg-zinc-800 aria-selected:text-zinc-50"
              >
                {opt.label}
              </CommandItem>
            ))}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}

function AnimeCard({ anime }: { anime: Anime }) {
  const malLink = `https://myanimelist.net/anime/${anime.mal_id}`;

  return (
    <div className="group perspective h-[420px] cursor-pointer">
      {/*
        Добавили transform-style: preserve-3d и убедились, что hover
        работает на всем контейнере group
      */}
      <div className="relative w-full h-full transition-all duration-700 preserve-3d group-hover:rotate-y-180">

        {/* ЛИЦЕВАЯ СТОРОНА */}
        <div className="absolute inset-0 backface-hidden w-full h-full">
          <Card className="w-full h-full overflow-hidden border-zinc-800 bg-zinc-900 rounded-2xl border-[1px] p-0 shadow-lg">
            <div className="relative w-full h-full">
              <img
                src={anime.image_url || 'https://placehold.co/300x450/18181b/FFF?text=No+Cover'}
                alt={anime.title}
                className="w-full h-full object-cover grayscale-[20%] group-hover:grayscale-0 transition-all duration-500"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-zinc-950 via-zinc-950/40 to-transparent" />
              <div className="absolute bottom-0 p-4 w-full space-y-1">
                <h3 className="font-bold text-base text-zinc-100 line-clamp-1 drop-shadow-lg uppercase tracking-tight">
                  {anime.title}
                </h3>
                {/* Исправили text-[100px] на text-[10px] */}
                <div className="flex items-center gap-1 text-yellow-500 font-black italic">
                   <Star className="h-3 w-3 fill-current" />
                   <span className="text-[10px]">{anime.score || "N/A"}</span>
                </div>
              </div>
            </div>
          </Card>
        </div>

        {/* ОБРАТНАЯ СТОРОНА (BACK) */}
        <div className="absolute inset-0 backface-hidden rotate-y-180 w-full h-full">
          <Card className="w-full h-full bg-zinc-900 border-zinc-800 p-6 flex flex-col shadow-2xl rounded-2xl border-[1px]">

            {/* ВЕРХ: Заголовок и линия (не сжимаются) */}
            <div className="shrink-0 space-y-3">
              <h3 className="font-black text-lg text-zinc-100 leading-tight uppercase italic line-clamp-2">
                {anime.title}
              </h3>
              <div className="h-0.5 bg-zinc-700 w-8" />
            </div>

            {/* СЕРЕДИНА: Область скролла (занимает всё свободное место) */}
            {/* flex-1 + min-h-0 — это секретная комбинация, чтобы скролл заработал */}
            <div className="flex-1 min-h-0 py-4">
              <ScrollArea className="h-full w-full">
                <div className="pr-4">
                  <p className="text-xs leading-relaxed font-medium text-zinc-400 italic">
                    {anime.description || "No description provided."}
                  </p>
                </div>
              </ScrollArea>
            </div>

            {/* НИЗ: Кнопка (не сжимается) */}
            <Button asChild className="shrink-0 w-full bg-zinc-100 text-zinc-950 hover:bg-white rounded-xl font-bold h-11 transition-transform active:scale-95">
              <a href={malLink} target="_blank" rel="noopener noreferrer">OPEN ON MAL</a>
            </Button>

          </Card>
        </div>

      </div>
    </div>
  );
}