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
import { SlidersHorizontal, Check, ChevronsUpDown, Search, Star, Calendar, AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
    Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList,
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
    data: string[];
    selected: string[];
    setSelected: (val: string[]) => void;
    placeholder: string;
}

interface SimpleSelectProps {
    placeholder: string;
    options: FilterOption[];
    value: string;
    setValue: (val: string) => void;
}

// Заглушки на случай, если бэк не ответил
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

    // Загрузка фильтров из БД при старте
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

            if (response.status === 404) {
                setResults([]);
                return;
            }

            if (!response.ok) throw new Error("Search failed");
            const data: RecommendationResponse = await response.json();
            setResults(data.model_response || []);
        } catch (error) {
            console.error("Error:", error);
            setResults([]);
        } finally {
            setLoading(false);
            setHasSearched(true);
        }
    };

    return (
        <div className="min-h-screen bg-zinc-950 text-zinc-50 font-sans selection:bg-zinc-800">
            <main className="max-w-7xl mx-auto px-4 py-16 space-y-10">

                {/* HEADER */}
                <div className="space-y-4 text-center">
                    <h1 className="text-5xl md:text-7xl font-black tracking-tighter bg-gradient-to-b from-white to-zinc-500 bg-clip-text text-transparent uppercase">
                        AniReco
                    </h1>
                    <p className="text-zinc-500 text-lg font-medium italic">Neural Semantic Search Engine</p>
                </div>

                {/* SEARCH PANEL */}
                <section className="bg-zinc-900/40 border border-zinc-800/50 p-6 md:p-8 rounded-3xl backdrop-blur-xl space-y-8 shadow-2xl">

                    {/* ПЕРВЫЙ РЯД: 4 колонки фильтров */}
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                        <div className="space-y-1.5">
                            <label className="text-[10px] font-bold uppercase tracking-widest text-zinc-600 ml-1">Genres</label>
                            <MultiSelect
                                data={availableGenres.length > 0 ? availableGenres : GENRES_FALLBACK}
                                selected={selectedGenres}
                                setSelected={setSelectedGenres}
                                placeholder="All Genres"
                            />
                        </div>

                        <div className="space-y-1.5">
                            <label className="text-[10px] font-bold uppercase tracking-widest text-zinc-600 ml-1">Themes</label>
                            <MultiSelect
                                data={availableThemes.length > 0 ? availableThemes : THEMES_FALLBACK}
                                selected={selectedThemes}
                                setSelected={setSelectedThemes}
                                placeholder="All Themes"
                            />
                        </div>

                        <div className="space-y-1.5">
                            <label className="text-[10px] font-bold uppercase tracking-widest text-zinc-600 ml-1">Release Period</label>
                            <div className="flex items-center gap-2">
                                <div className="relative flex-1">
                                    <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-zinc-600" />
                                    <Input
                                        type="number"
                                        placeholder="From"
                                        value={yearMin}
                                        onChange={(e) => setYearMin(e.target.value)}
                                        className={cn("bg-zinc-950 border-zinc-800 pl-9 h-11 rounded-xl focus-visible:ring-zinc-700", noSpinners)}
                                    />
                                </div>
                                <div className="relative flex-1">
                                    <Input
                                        type="number"
                                        placeholder="To"
                                        value={yearMax}
                                        onChange={(e) => setYearMax(e.target.value)}
                                        className={cn("bg-zinc-950 border-zinc-800 h-11 rounded-xl focus-visible:ring-zinc-700", noSpinners)}
                                    />
                                </div>
                            </div>
                        </div>

                        <div className="space-y-1.5">
                            <label className="text-[10px] font-bold uppercase tracking-widest text-zinc-600 ml-1">Format</label>
                            <SimpleSelect
                                placeholder="All Formats"
                                options={TYPES_OPTIONS}
                                value={type}
                                setValue={setType}
                            />
                        </div>
                    </div>

                    {/* ВТОРОЙ РЯД: ЕДИНАЯ ПАНЕЛЬ ПОИСКА */}
                    <div className="relative flex items-center w-full bg-zinc-950 border border-zinc-800 rounded-2xl focus-within:ring-1 focus-within:ring-zinc-700 transition-all shadow-inner p-2 pl-4">

                        {/* 1. Иконка поиска */}
                        <Search className="h-5 w-5 text-zinc-600 shrink-0" />

                        {/* 2. Поле ввода */}
                        <Input
                            placeholder="Describe vibe: 'story about a silent hero in a magical forest'..."
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                            className="flex-1 bg-transparent border-none text-lg h-14 focus-visible:ring-0 placeholder:text-zinc-700 px-4"
                        />

                        {/* 3. Правая часть: Рейтинг + Разделитель + Сортировка + Кнопка */}
                        <div className="flex items-center gap-3 shrink-0">

                            {/* Рейтинг (инпут) */}
                            <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-zinc-900/50 border border-zinc-800/50">
                                <Star className="h-3.5 w-3.5 text-yellow-500 fill-yellow-500/20" />
                                <input
                                    type="number"
                                    step="0.1"
                                    min="0"
                                    max="10"
                                    placeholder="0.0"
                                    value={minScore}
                                    onChange={(e) => setMinScore(e.target.value)}
                                    className={cn("w-8 bg-transparent border-none text-xs font-bold focus:outline-none text-yellow-500 placeholder:text-zinc-800", noSpinners)}
                                />
                            </div>

                            {/* Вертикальная черта */}
                            <div className="h-8 w-px bg-zinc-800 mx-1 hidden sm:block" />

                            {/* ИКОНКА СОРТИРОВКИ */}
                            <SortDropdown
                                value={sortBy}
                                setValue={setSortBy}
                                includeAdult={includeAdult}
                                setIncludeAdult={setIncludeAdult}
                            />

                            {/* Кнопка "Найти" */}
                            <Button
                                onClick={handleSearch}
                                disabled={loading}
                                className="h-12 px-8 bg-zinc-50 text-zinc-950 hover:bg-white font-black rounded-xl transition-all active:scale-95 shadow-lg ml-1"
                            >
                                {loading ? "..." : "FIND"}
                            </Button>
                        </div>
                    </div>

                    {/* СТАРЫЙ ЧЕКБОКС УДАЛЕН ОТСЮДА */}
                </section>

                {/* БЛОК ПРЕДУПРЕЖДЕНИЯ (ALERT) */}
                {!loading && hasSearched && results.length === 0 && (
                    <div className="max-w-2xl mx-auto">
                        <Alert className="bg-yellow-500/10 border-yellow-500/20 text-yellow-200 shadow-lg">
                            <AlertCircle className="h-5 w-5 stroke-yellow-500" />
                            <AlertTitle className="text-yellow-500 font-bold ml-2">Nothing found</AlertTitle>
                            <AlertDescription className="text-zinc-400 ml-2 mt-1">
                                We couldn't find any anime matching these filters. Try lowering the rating or removing
                                some genres.
                            </AlertDescription>
                        </Alert>
                    </div>
                )}

                {/* RESULTS */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                    {results.map((anime) => (
                        <AnimeCard key={anime.mal_id} anime={anime} />
                    ))}
                </div>

                {!loading && !hasSearched && results.length === 0 && (
                    <div className="text-center py-20 border-2 border-dashed border-zinc-900 rounded-3xl">
                        <p className="text-zinc-600 font-medium italic">Your next favorite story is just one search away.</p>
                    </div>
                )}
            </main>
        </div>
    );
}

// --- UI КОМПОНЕНТЫ ---

function MultiSelect({ data, selected, setSelected, placeholder }: MultiSelectProps) {
    const [open, setOpen] = useState(false);
    const toggle = (val: string) => {
        setSelected(selected.includes(val) ? selected.filter((i: any) => i !== val) : [...selected, val]);
    };

    return (
        <Popover open={open} onOpenChange={setOpen}>
            <PopoverTrigger asChild>
                <Button variant="outline" className="w-full justify-between h-11 bg-zinc-950 border-zinc-800 rounded-xl text-zinc-400 font-normal">
                    <span className="truncate">{selected.length > 0 ? `${selected.length} selected` : placeholder}</span>
                    <ChevronsUpDown className="h-4 w-4 opacity-50" />
                </Button>
            </PopoverTrigger>
            <PopoverContent className="w-[200px] p-0 bg-zinc-950 border-zinc-800 shadow-2xl">
                <Command className="bg-zinc-950">
                    <CommandInput placeholder="Search..." className="text-zinc-200" />
                    <CommandList>
                        <CommandEmpty>No results.</CommandEmpty>
                        <CommandGroup className="max-h-60 overflow-y-auto">
                            {data.map((item: string) => (
                                <CommandItem key={item} onSelect={() => toggle(item)}
                                    className="cursor-pointer text-zinc-400 aria-selected:bg-zinc-800 aria-selected:text-zinc-50">
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
                                onSelect={() => {
                                    setValue(opt.value === value ? "" : opt.value);
                                    setOpen(false);
                                }}
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
                                <div className="flex items-center gap-1 text-yellow-500 font-black italic">
                                    <Star className="h-3 w-3 fill-current" />
                                    <span className="text-[10px]">{anime.score || "N/A"}</span>
                                </div>
                            </div>
                        </div>
                    </Card>
                </div>

                {/* ОБРАТНАЯ СТОРОНА */}
                <div className="absolute inset-0 backface-hidden rotate-y-180 w-full h-full">
                    <Card className="w-full h-full bg-zinc-900 border-zinc-800 p-6 flex flex-col shadow-2xl rounded-2xl border-[1px]">
                        <div className="shrink-0 space-y-3">
                            <h3 className="font-black text-lg text-zinc-100 leading-tight uppercase italic line-clamp-2">
                                {anime.title}
                            </h3>
                            <div className="h-0.5 bg-zinc-700 w-8" />
                        </div>
                        <div className="flex-1 min-h-0 py-4">
                            <ScrollArea className="h-full w-full">
                                <div className="pr-4">
                                    <p className="text-xs leading-relaxed font-medium text-zinc-400 italic">
                                        {anime.description || "No description provided."}
                                    </p>
                                </div>
                            </ScrollArea>
                        </div>
                        <Button asChild className="shrink-0 w-full bg-zinc-100 text-zinc-950 hover:bg-white rounded-xl font-bold h-11 transition-transform active:scale-95">
                            <a href={malLink} target="_blank" rel="noopener noreferrer">OPEN ON MAL</a>
                        </Button>
                    </Card>
                </div>
            </div>
        </div>
    );
}

function SortDropdown({
    value,
    setValue,
    includeAdult,
    setIncludeAdult
}: {
    value: string,
    setValue: (v: string) => void,
    includeAdult: boolean,
    setIncludeAdult: (v: boolean) => void
}) {
    const labels: Record<string, string> = {
        relevance: "Relevance",
        rating: "Top Rated",
        popularity: "Most Popular"
    };

    return (
        <DropdownMenu>
            <DropdownMenuTrigger asChild>
                <Button
                    variant="ghost"
                    size="icon"
                    className="h-10 w-10 rounded-xl border border-transparent hover:border-zinc-700 hover:bg-zinc-800 data-[state=open]:bg-zinc-800 data-[state=open]:border-zinc-600 data-[state=open]:text-zinc-50 transition-all text-zinc-500"
                >
                    <SlidersHorizontal className="h-4 w-4" />
                </Button>
            </DropdownMenuTrigger>

            <DropdownMenuContent className="w-56 bg-zinc-950 border-zinc-800 shadow-xl mr-4" align="end">
                <DropdownMenuLabel className="text-[10px] font-bold uppercase tracking-widest text-zinc-500">
                    Sort Results By
                </DropdownMenuLabel>

                <DropdownMenuSeparator className="bg-zinc-800" />

                <DropdownMenuRadioGroup value={value} onValueChange={setValue}>
                    {Object.entries(labels).map(([key, label]) => (
                        <DropdownMenuRadioItem
                            key={key}
                            value={key}
                            className="cursor-pointer text-zinc-400 focus:text-zinc-50 focus:bg-zinc-900"
                        >
                            {label}
                        </DropdownMenuRadioItem>
                    ))}
                </DropdownMenuRadioGroup>

                <DropdownMenuSeparator className="bg-zinc-800 my-1" />

                <div
                    className="flex items-center gap-3 p-2 rounded-sm hover:bg-zinc-900 cursor-pointer transition-colors outline-none select-none"
                    onClick={(e) => {
                        e.preventDefault();
                        setIncludeAdult(!includeAdult);
                    }}
                >
                    <Checkbox
                        checked={includeAdult}
                        onCheckedChange={(c) => setIncludeAdult(!!c)}
                        className="h-4 w-4 border-zinc-600 data-[state=checked]:bg-white data-[state=checked]:border-white data-[state=checked]:text-black transition-all shadow-sm"
                    />
                    <span className="text-sm text-zinc-400 font-medium">
                        Include 18+ Content
                    </span>
                </div>

            </DropdownMenuContent>
        </DropdownMenu>
    );
}