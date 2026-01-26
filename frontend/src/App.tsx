import { useState } from 'react';
import { Check, ChevronsUpDown, Search } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import {
  Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList,
} from "@/components/ui/command";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";

// --- 1. ОПРЕДЕЛЯЕМ ТИПЫ (INTERFACES) ---

// Тип для одного варианта фильтра (например, {value: "action", label: "Action"})
interface FilterOption {
  value: string;
  label: string;
}

// Тип для объекта аниме, который приходит с бэкенда
interface Anime {
  mal_id: number;
  title: string;
  description: string;
  score?: number;
  image_url?: string;
  status?: string;
}

// Тип для пропсов компонента фильтра
interface FilterComboboxProps {
  title: string;
  options: FilterOption[];
  value: string;
  setValue: (value: string) => void;
}

// Тип для пропсов карточки
interface AnimeCardProps {
  anime: Anime;
}

// Тип ответа от API
interface RecommendationResponse {
  model_response: Anime[];
}

// --- 2. ДАННЫЕ И ХЕЛПЕРЫ ---

const filterOptions = {
  genres: [
    { value: "Action", label: "Action" },
    { value: "Adventure", label: "Adventure" },
    { value: "Comedy", label: "Comedy" },
    { value: "Drama", label: "Drama" },
    { value: "Sci-Fi", label: "Sci-Fi" },
    { value: "Slice of Life", label: "Slice of Life" },
    { value: "Fantasy", label: "Fantasy" },
    { value: "Romance", label: "Romance" },
    { value: "Mystery", label: "Mystery" },
    { value: "Horror", label: "Horror" },
    { value: "Sports", label: "Sports" },
    { value: "Supernatural", label: "Supernatural" },
  ],
  types: [
    { value: "tv", label: "TV Series" },
    { value: "movie", label: "Movie" },
    { value: "ova", label: "OVA" },
    { value: "special", label: "Special" }
  ],
  periods: [
    { value: "2020s", label: "2020s (Modern)" },
    { value: "2010s", label: "2010s" },
    { value: "2000s", label: "2000s" },
    { value: "1990s", label: "1990s" },
    { value: "old",   label: "Old School (Pre-90s)" },
  ]
};

const getYearRange = (periodValue: string) => {
  switch (periodValue) {
    case "2020s": return { min: 2020, max: 2029 };
    case "2010s": return { min: 2010, max: 2019 };
    case "2000s": return { min: 2000, max: 2009 };
    case "1990s": return { min: 1990, max: 1999 };
    case "old":   return { min: 1900, max: 1989 };
    default:      return { min: null, max: null };
  }
};

// --- 3. ГЛАВНЫЙ КОМПОНЕНТ ---

export default function AnimeApp() {
  const [query, setQuery] = useState("");
  const [genre, setGenre] = useState("");
  const [rating, setRating] = useState(""); // Используем как "Period"
  const [type, setType] = useState("");

  // Явно указываем, что results - это массив объектов Anime
  const [results, setResults] = useState<Anime[]>([]);
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    if (!query && !genre && !type && !rating) return;
    setLoading(true);

    const { min, max } = getYearRange(rating);

    try {
      const response = await fetch('http://127.0.0.1:8000/recommend', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text_query: query,
          genre: genre || null,
          type: type || null,
          year_min: min,
          year_max: max
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        console.error("Error:", errorData);
        return;
      }

      const data: RecommendationResponse = await response.json();
      setResults(data.model_response);

    } catch (error) {
      console.error("Network Error:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-50 font-sans selection:bg-zinc-800">
      <main className="max-w-6xl mx-auto px-4 py-16 space-y-12">
        <div className="space-y-4 text-center">
          <h1 className="text-4xl md:text-6xl font-extrabold tracking-tight bg-gradient-to-b from-zinc-50 to-zinc-400 bg-clip-text text-transparent">
            AniReco
          </h1>
          <p className="text-zinc-400 text-lg max-w-2xl mx-auto">
            AI-powered anime recommendation system. <br/>
            Search by plot, vibe, or semantic description.
          </p>
        </div>

        <section className="bg-zinc-900/50 border border-zinc-800 p-6 rounded-2xl backdrop-blur-sm space-y-6 shadow-2xl">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <FilterCombobox
              title="Genre"
              options={filterOptions.genres}
              value={genre}
              setValue={setGenre}
            />
            <FilterCombobox
              title="Period"
              options={filterOptions.periods}
              value={rating}
              setValue={setRating}
            />
            <FilterCombobox
              title="Type"
              options={filterOptions.types}
              value={type}
              setValue={setType}
            />
          </div>

          <div className="relative flex items-center group">
            <Search className="absolute left-4 h-5 w-5 text-zinc-500 group-focus-within:text-zinc-200 transition-colors" />
            <Input
              placeholder="Describe what you want to watch (e.g. 'Cyberpunk city with a dark atmosphere')..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              className="pl-12 h-14 bg-zinc-950/50 border-zinc-800 text-lg rounded-xl focus-visible:ring-zinc-700 transition-all"
            />
            <Button
              onClick={handleSearch}
              disabled={loading}
              className="absolute right-2 h-10 px-6 bg-zinc-50 text-zinc-950 hover:bg-zinc-200 font-bold rounded-lg transition-all"
            >
              {loading ? "Thinking..." : "Find"}
            </Button>
          </div>
        </section>

        {results.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-8">
            {results.map((anime) => (
              <AnimeCard key={anime.mal_id} anime={anime} />
            ))}
          </div>
        )}

        {results.length === 0 && !loading && (
           <div className="text-center text-zinc-600 mt-10">
             Try searching for something or apply filters.
           </div>
        )}
      </main>
    </div>
  );
}

// --- КОМПОНЕНТЫ UI (Теперь типизированные) ---

function FilterCombobox({ title, options, value, setValue }: FilterComboboxProps) {
  const [open, setOpen] = useState(false);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          className="w-full justify-between h-11 bg-zinc-900 border-zinc-800 hover:bg-zinc-800 hover:text-zinc-50 transition-all rounded-xl text-zinc-400 font-normal"
        >
          {value
            ? options.find((o) => o.value === value)?.label
            : `Select ${title}`}
          <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[200px] p-0 bg-zinc-950 border-zinc-800 shadow-2xl">
        <Command className="bg-zinc-950 text-zinc-200">
          <CommandInput placeholder={`Search ${title}...`} className="h-9 border-none focus:ring-0" />
          <CommandList className="border-t border-zinc-800">
            <CommandEmpty className="py-2 px-4 text-xs text-zinc-500">No results.</CommandEmpty>
            <CommandGroup>
              {options.map((opt) => (
                <CommandItem
                  key={opt.value}
                  value={opt.value}
                  onSelect={(currentValue) => {
                    setValue(currentValue === value ? "" : opt.value);
                    setOpen(false);
                  }}
                  className="cursor-pointer flex items-center px-2 py-2 text-sm text-zinc-400 aria-selected:bg-zinc-800 aria-selected:text-zinc-50 hover:bg-zinc-800 hover:text-zinc-50"
                >
                  <Check
                    className={cn(
                      "mr-2 h-4 w-4 text-white",
                      value === opt.value ? "opacity-100" : "opacity-0"
                    )}
                  />
                  {opt.label}
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}

function AnimeCard({ anime }: AnimeCardProps) {
  const malLink = `https://myanimelist.net/anime/${anime.mal_id}`;

  return (
    <div className="group perspective w-full h-[450px] cursor-pointer">
      <div className="relative w-full h-full transition-all duration-700 preserve-3d group-hover:rotate-y-180">
        {/* FRONT */}
        <div className="absolute inset-0 backface-hidden">
          <Card className="w-full h-full overflow-hidden border-zinc-800 bg-zinc-950 rounded-2xl border-[1px] p-0 shadow-2xl">
            <div className="relative w-full h-full">
              <img
                src={anime.image_url || 'https://placehold.co/300x450/18181b/FFF?text=No+Image'}
                alt={anime.title}
                className="absolute inset-0 w-full h-full object-cover object-center transition-transform duration-700 group-hover:scale-110"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-zinc-950 via-zinc-950/20 to-transparent opacity-90" />
              <div className="absolute bottom-0 left-0 right-0 p-5 space-y-2">
                <h3 className="text-white font-bold text-lg leading-tight tracking-tight drop-shadow-md line-clamp-2">
                  {anime.title}
                </h3>
                {anime.score && (
                  <div className="flex items-center gap-1.5 text-yellow-500 font-bold text-xs uppercase tracking-tighter">
                    <span>★</span>
                    <span>{anime.score}</span>
                  </div>
                )}
              </div>
            </div>
          </Card>
        </div>

        {/* BACK */}
        <div className="absolute inset-0 backface-hidden rotate-y-180">
          <Card className="w-full h-full bg-zinc-900 border-zinc-700 p-6 flex flex-col justify-between shadow-2xl rounded-2xl border-2">
            <div className="space-y-4 overflow-hidden h-full flex flex-col">
              <div className="space-y-1 shrink-0">
                <h3 className="font-bold text-lg text-zinc-50 tracking-tight line-clamp-2 uppercase">
                  {anime.title}
                </h3>
                {anime.status && (
                  <p className="text-[10px] text-zinc-500 font-bold uppercase tracking-widest">
                    {anime.status}
                  </p>
                )}
              </div>
              <div className="h-px bg-zinc-800 w-full shrink-0" />

              {/* Прокрутка описания */}
              <div className="flex-1 overflow-y-auto pr-1 custom-scrollbar">
                 <p className="text-sm text-zinc-400 leading-relaxed font-light italic">
                  {anime.description || "No description available."}
                </p>
              </div>
            </div>

            <Button asChild className="w-full bg-zinc-50 text-zinc-950 hover:bg-zinc-200 rounded-xl font-bold h-12 mt-4 shrink-0 transition-all active:scale-95">
              <a href={malLink} target="_blank" rel="noopener noreferrer">
                MyAnimeList
              </a>
            </Button>
          </Card>
        </div>
      </div>
    </div>
  );
}