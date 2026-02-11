import { useEffect, useState } from 'react';
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuLabel,
    DropdownMenuRadioGroup,
    DropdownMenuRadioItem,
    DropdownMenuSeparator,
    DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { SlidersHorizontal, Check, ChevronsUpDown, Search, Star, AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
    Command, CommandGroup, CommandInput, CommandItem, CommandList,
} from "@/components/ui/command";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Checkbox } from "@/components/ui/checkbox";

const noSpinners = "[appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none";

// --- 1. ТИПЫ ---

interface Anime {
    mal_id: number;
    title: string;
    description: string;
    score?: number;
    image_url?: string;
}

interface FilterOption {
    value: string;
    label: string;
}

interface RecommendationResponse {
    model_response: Anime[];
}

const GENRES_FALLBACK = ["Action", "Adventure", "Comedy", "Drama", "Sci-Fi", "Fantasy", "Romance"];
const THEMES_FALLBACK = ["Gore", "Military", "Music", "Psychological", "School", "Space"];

const TYPES_OPTIONS: FilterOption[] = [
    { value: "TV", label: "TV Series" },
    { value: "Movie", label: "Movie" },
    { value: "OVA", label: "OVA" },
    { value: "ONA", label: "ONA" },
];

// --- 2. ГЛАВНЫЙ КОМПОНЕНТ ---

export default function AnimeApp() {
    const [hasSearched, setHasSearched] = useState(false);
    const [query, setQuery] = useState("");
    const [results, setResults] = useState<Anime[]>([]);
    const [loading, setLoading] = useState(false);

    const [availableGenres, setAvailableGenres] = useState<string[]>([]);
    const [availableThemes, setAvailableThemes] = useState<string[]>([]);

    const [selectedGenres, setSelectedGenres] = useState<string[]>([]);
    const [selectedThemes, setSelectedThemes] = useState<string[]>([]);

    const [yearMin, setYearMin] = useState<string>("");
    const [yearMax, setYearMax] = useState<string>("");
    const [minScore, setMinScore] = useState<string>("");

    const [type, setType] = useState("");
    const [sortBy, setSortBy] = useState("relevance");

    const [includeAdult, setIncludeAdult] = useState(false);

    useEffect(() => {
        fetch('/filters')
            .then(res => res.json())
            .then(data => {
                if (data.genres && data.genres.length > 0) setAvailableGenres(data.genres);
                if (data.themes && data.themes.length > 0) setAvailableThemes(data.themes);
            })
            .catch(err => console.error("FETCH ERROR:", err));
    }, []);

    const handleSearch = async () => {
        if (!query.trim() && selectedGenres.length === 0 && selectedThemes.length === 0 && !type && !yearMin && !yearMax && !minScore) return;
        setLoading(true);
        setHasSearched(false);
        try {
            const response = await fetch('/recommend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text_query: query,
                    genres: selectedGenres.length > 0 ? selectedGenres : null,
                    themes: selectedThemes.length > 0 ? selectedThemes : null,
                    type: type || null,
                    year_min: yearMin ? parseInt(yearMin) : null,
                    year_max: yearMax ? parseInt(yearMax) : null,
                    min_score: minScore ? parseFloat(minScore) : null,
                    sort_by: sortBy,
                    include_adult: includeAdult
                }),
            });
            if (!response.ok) throw new Error();
            const data: RecommendationResponse = await response.json();
            setResults(data.model_response || []);
        } catch {
            setResults([]);
        } finally {
            setLoading(false);
            setHasSearched(true);
        }
    };

    return (
        <div className="min-h-screen bg-black text-zinc-50 font-sans antialiased selection:bg-zinc-800">
            <main className="max-w-7xl mx-auto px-4 py-16 space-y-12">

                {/* HEADER */}
                <div className="space-y-4 text-center">
                    <h1 className="text-5xl md:text-7xl font-bold tracking-tighter text-white uppercase">
                        AniReco
                    </h1>
                    <p className="text-zinc-400 text-lg font-normal">Neural Semantic Search Engine</p>
                </div>

                {/* SEARCH PANEL */}
                <section className="bg-[#09090b] border border-zinc-800 p-6 md:p-8 rounded-[32px] shadow-2xl space-y-8">

                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
                        <div className="space-y-2">
                            <label className="text-[11px] font-bold uppercase tracking-wider text-zinc-500 ml-3">Genres</label>
                            <MultiSelect
                                data={availableGenres.length > 0 ? availableGenres : GENRES_FALLBACK}
                                selected={selectedGenres}
                                setSelected={setSelectedGenres}
                                placeholder="Select Genres"
                            />
                        </div>

                        <div className="space-y-2">
                            <label className="text-[11px] font-bold uppercase tracking-wider text-zinc-500 ml-3">Themes</label>
                            <MultiSelect
                                data={availableThemes.length > 0 ? availableThemes : THEMES_FALLBACK}
                                selected={selectedThemes}
                                setSelected={setSelectedThemes}
                                placeholder="Select Themes"
                            />
                        </div>

                        <div className="space-y-2">
                            <label className="text-[11px] font-bold uppercase tracking-wider text-zinc-500 ml-3">Release Period</label>
                            <div className="flex items-center gap-2">
                                <Input
                                    type="number"
                                    placeholder="From"
                                    value={yearMin}
                                    onChange={(e) => setYearMin(e.target.value)}
                                    className={cn("bg-zinc-950 border-zinc-800 h-11 rounded-2xl focus-visible:ring-zinc-700 placeholder:text-zinc-600 text-zinc-200 px-4", noSpinners)}
                                />
                                <Input
                                    type="number"
                                    placeholder="To"
                                    value={yearMax}
                                    onChange={(e) => setYearMax(e.target.value)}
                                    className={cn("bg-zinc-950 border-zinc-800 h-11 rounded-2xl focus-visible:ring-zinc-700 placeholder:text-zinc-600 text-zinc-200 px-4", noSpinners)}
                                />
                            </div>
                        </div>

                        <div className="space-y-2">
                            <label className="text-[11px] font-bold uppercase tracking-wider text-zinc-500 ml-3">Format</label>
                            <SimpleSelect
                                placeholder="Any Format"
                                options={TYPES_OPTIONS}
                                value={type}
                                setValue={setType}
                            />
                        </div>
                    </div>

                    <div className="h-px w-full bg-zinc-800" />

                    <div className="relative flex items-center w-full bg-zinc-950 border border-zinc-800 rounded-2xl focus-within:ring-1 focus-within:ring-zinc-600 transition-all p-2 pl-4">
                        <Search className="h-5 w-5 text-zinc-500 shrink-0 mr-3" />
                        <Input
                            placeholder="Describe vibe: 'story about a silent hero in a magical forest'..."
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                            className="flex-1 bg-transparent border-none text-base h-12 focus-visible:ring-0 placeholder:text-zinc-600 text-zinc-200 px-0"
                        />
                        <div className="flex items-center gap-2 shrink-0 ml-2">
                            <div className="hidden sm:flex items-center gap-2 px-3 py-2 rounded-xl bg-[#09090b] border border-zinc-800">
                                <Star className="h-4 w-4 text-yellow-500 fill-yellow-500" />
                                <input
                                    type="number"
                                    step="0.1"
                                    min="0"
                                    max="10"
                                    placeholder="0.0"
                                    value={minScore}
                                    onChange={(e) => setMinScore(e.target.value)}
                                    className={cn("w-8 bg-transparent border-none text-sm font-semibold focus:outline-none text-zinc-200 placeholder:text-zinc-600", noSpinners)}
                                />
                            </div>
                            <div className="h-8 w-px bg-zinc-800 mx-2 hidden sm:block" />
                            <SortDropdown
                                value={sortBy}
                                setValue={setSortBy}
                                includeAdult={includeAdult}
                                setIncludeAdult={setIncludeAdult}
                            />
                            <Button
                                onClick={handleSearch}
                                disabled={loading}
                                className="h-12 px-8 bg-white text-black hover:bg-zinc-200 font-bold rounded-xl transition-all active:scale-95 ml-1"
                            >
                                {loading ? "..." : "Find"}
                            </Button>
                        </div>
                    </div>
                </section>

                {/* ALERT */}
                {!loading && hasSearched && results.length === 0 && (
                    <div className="max-w-2xl mx-auto">
                        <Alert className="bg-[#09090b] border-zinc-800 text-zinc-200">
                            <AlertCircle className="h-5 w-5 stroke-zinc-400" />
                            <AlertTitle className="text-zinc-100 font-bold ml-2">Nothing found</AlertTitle>
                            <AlertDescription className="text-zinc-400 ml-2 mt-1">
                                We couldn't find any anime matching these filters. Try removing some genres.
                            </AlertDescription>
                        </Alert>
                    </div>
                )}

                {/* RESULTS */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-x-6 gap-y-10">
                    {results.map((anime) => (
                        <AnimeCard key={anime.mal_id} anime={anime} />
                    ))}
                </div>
            </main>
        </div>
    );
}

// --- UI КОМПОНЕНТЫ (ИСПРАВЛЕНЫ ОТСТУПЫ ВЫДЕЛЕНИЯ) ---

function MultiSelect({ data, selected, setSelected, placeholder }: { data: string[], selected: string[], setSelected: any, placeholder: string }) {
    const [open, setOpen] = useState(false);
    return (
        <Popover open={open} onOpenChange={setOpen}>
            <PopoverTrigger asChild>
                <Button variant="outline" className="w-full justify-between h-11 bg-zinc-950 border-zinc-800 rounded-2xl text-zinc-300 font-normal hover:bg-zinc-900 hover:text-zinc-100 px-4">
                    <span className="truncate">{selected.length > 0 ? `${selected.length} selected` : placeholder}</span>
                    <ChevronsUpDown className="h-4 w-4 opacity-30" />
                </Button>
            </PopoverTrigger>
            <PopoverContent className="w-[200px] p-0 bg-[#09090b] border-zinc-800 shadow-xl rounded-xl overflow-hidden" align="start">
                <Command className="bg-transparent">
                    <CommandInput placeholder="Search..." className="text-zinc-200" />
                    <CommandList>
                        <CommandGroup className="max-h-60 overflow-y-auto p-1">
                            {data.map((item) => (
                                <CommandItem
                                    key={item}
                                    onSelect={() => setSelected(selected.includes(item) ? selected.filter(i => i !== item) : [...selected, item])}
                                    className="cursor-pointer text-zinc-400 aria-selected:bg-zinc-100 aria-selected:text-black rounded-md px-2 py-1.5 transition-colors"
                                >
                                    <Check className={cn("mr-2 h-4 w-4", selected.includes(item) ? "opacity-100" : "opacity-0")} />
                                    {item}
                                </CommandItem>
                            ))}
                        </CommandGroup>
                    </CommandList>
                </Command>
            </PopoverContent>
        </Popover>
    );
}

function SimpleSelect({ placeholder, options, value, setValue }: { placeholder: string, options: any[], value: string, setValue: any }) {
    const [open, setOpen] = useState(false);
    return (
        <Popover open={open} onOpenChange={setOpen}>
            <PopoverTrigger asChild>
                <Button variant="outline" className="w-full justify-between h-11 bg-zinc-950 border-zinc-800 rounded-2xl text-zinc-300 font-normal hover:bg-zinc-900 hover:text-zinc-100 px-4">
                    {value ? options.find((o) => o.value === value)?.label : placeholder}
                    <ChevronsUpDown className="h-4 w-4 opacity-30" />
                </Button>
            </PopoverTrigger>
            <PopoverContent className="w-[200px] p-0 bg-[#09090b] border-zinc-800 shadow-xl rounded-xl overflow-hidden" align="start">
                <Command className="bg-transparent">
                    <CommandList>
                        <CommandGroup className="p-1">
                            {options.map((opt) => (
                                <CommandItem
                                    key={opt.value}
                                    onSelect={() => { setValue(opt.value === value ? "" : opt.value); setOpen(false); }}
                                    className="cursor-pointer text-zinc-400 aria-selected:bg-zinc-100 aria-selected:text-black rounded-md px-2 py-1.5 transition-colors"
                                >
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

function AnimeCard({ anime }: { anime: Anime }) {
    return (
        <div className="group perspective h-[420px] cursor-pointer">
            <div className="relative w-full h-full transition-all duration-500 preserve-3d group-hover:rotate-y-180">
                {/* ЛИЦЕВАЯ СТОРОНА */}
                <div className="absolute inset-0 backface-hidden w-full h-full">
                    <Card className="w-full h-full overflow-hidden border-zinc-800 bg-[#09090b] rounded-2xl border p-0 shadow-lg group-hover:shadow-zinc-900/50">
                        <div className="relative w-full h-full">
                            <img src={anime.image_url} alt="" className="w-full h-full object-cover grayscale-[40%] group-hover:grayscale-0 transition-all duration-500" />
                            <div className="absolute inset-0 bg-gradient-to-t from-black via-black/50 to-transparent opacity-80" />
                            <div className="absolute bottom-0 p-5 w-full space-y-1">
                                <h3 className="font-bold text-lg text-white line-clamp-2 leading-tight">{anime.title}</h3>
                                <div className="flex items-center gap-1 bg-white/10 w-fit px-2 py-0.5 rounded text-white font-bold text-xs backdrop-blur-md">
                                    <Star className="h-3 w-3 text-yellow-500 fill-yellow-500" />
                                    <span>{anime.score || "N/A"}</span>
                                </div>
                            </div>
                        </div>
                    </Card>
                </div>
                {/* ОБРАТНАЯ СТОРОНА */}
                <div className="absolute inset-0 backface-hidden rotate-y-180 w-full h-full">
                    <Card className="w-full h-full bg-[#09090b] border-zinc-800 p-6 flex flex-col rounded-2xl border">
                        <h3 className="font-bold text-lg text-white leading-tight mb-3">{anime.title}</h3>
                        <div className="h-px bg-zinc-800 w-full mb-4" />
                        <ScrollArea className="flex-1 min-h-0 pr-4">
                            <p className="text-sm leading-relaxed text-zinc-400 italic">{anime.description || "No description provided."}</p>
                        </ScrollArea>
                        <Button asChild className="mt-4 w-full bg-white text-black hover:bg-zinc-200 rounded-xl font-semibold h-10">
                            <a href={`https://myanimelist.net/anime/${anime.mal_id}`} target="_blank" rel="noreferrer">View on MyAnimeList</a>
                        </Button>
                    </Card>
                </div>
            </div>
        </div>
    );
}

function SortDropdown({ value, setValue, includeAdult, setIncludeAdult }: any) {
    return (
        <DropdownMenu>
            <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon" className="h-12 w-12 rounded-xl text-zinc-400 hover:text-white hover:bg-zinc-800">
                    <SlidersHorizontal className="h-5 w-5" />
                </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent className="w-56 bg-[#09090b] border-zinc-800 shadow-2xl mr-4 rounded-xl p-1" align="end">
                <DropdownMenuLabel className="text-[10px] font-bold uppercase tracking-widest text-zinc-500 px-3 py-2">Sort Results By</DropdownMenuLabel>
                <DropdownMenuSeparator className="bg-zinc-800 mx-1" />
                <DropdownMenuRadioGroup value={value} onValueChange={setValue}>
                    <DropdownMenuRadioItem value="relevance" className="cursor-pointer text-zinc-400 focus:text-white focus:bg-zinc-900 rounded-md">Relevance</DropdownMenuRadioItem>
                    <DropdownMenuRadioItem value="rating" className="cursor-pointer text-zinc-400 focus:text-white focus:bg-zinc-900 rounded-md">Top Rated</DropdownMenuRadioItem>
                    <DropdownMenuRadioItem value="popularity" className="cursor-pointer text-zinc-400 focus:text-white focus:bg-zinc-900 rounded-md">Most Popular</DropdownMenuRadioItem>
                </DropdownMenuRadioGroup>
                <DropdownMenuSeparator className="bg-zinc-800 mx-1 my-1" />
                <div className="flex items-center gap-3 p-2 rounded-md hover:bg-zinc-900 cursor-pointer transition-colors group" onClick={(e) => { e.preventDefault(); setIncludeAdult(!includeAdult); }}>
                    <Checkbox checked={includeAdult} className="h-4 w-4 border-zinc-600 data-[state=checked]:bg-white data-[state=checked]:border-white data-[state=checked]:text-black transition-all" />
                    <span className="text-sm text-zinc-400 group-hover:text-white transition-colors">Include 18+ Content</span>
                </div>
            </DropdownMenuContent>
        </DropdownMenu>
    );
}