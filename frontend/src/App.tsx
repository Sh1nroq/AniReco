import {useEffect, useState, useCallback} from 'react';
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuLabel,
    DropdownMenuRadioGroup,
    DropdownMenuRadioItem,
    DropdownMenuSeparator,
    DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {SlidersHorizontal, Check, ChevronsUpDown, Search, Star, AlertCircle, X, RotateCcw} from "lucide-react";
import {cn} from "@/lib/utils";
import {Button} from "@/components/ui/button";
import {Input} from "@/components/ui/input";
import {Card} from "@/components/ui/card";
import {Alert, AlertDescription, AlertTitle} from "@/components/ui/alert";
import {
    Command, CommandGroup, CommandInput, CommandItem, CommandList,
} from "@/components/ui/command";
import {Popover, PopoverContent, PopoverTrigger} from "@/components/ui/popover";
import {ScrollArea} from "@/components/ui/scroll-area";
import {Checkbox} from "@/components/ui/checkbox";

const noSpinners = "[appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none";

const CURRENT_YEAR = new Date().getFullYear();
const ANIME_FIRST_YEAR = 1917;

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

interface ValidationErrors {
    yearMin?: string;
    yearMax?: string;
    minScore?: string;
    yearRange?: string;
}

const GENRES_FALLBACK = ["Action", "Adventure", "Comedy", "Drama", "Sci-Fi", "Fantasy", "Romance"];
const THEMES_FALLBACK = ["Gore", "Military", "Music", "Psychological", "School", "Space"];

const TYPES_OPTIONS: FilterOption[] = [
    {value: "TV", label: "TV Series"},
    {value: "Movie", label: "Movie"},
    {value: "OVA", label: "OVA"},
    {value: "ONA", label: "ONA"},
];

// --- 2. ХЕЛПЕРЫ ВАЛИДАЦИИ ---

function validateYear(value: string): string | undefined {
    if (!value) return undefined;
    const num = parseInt(value);
    if (isNaN(num)) return "Must be a number";
    if (num < ANIME_FIRST_YEAR) return `Min year is ${ANIME_FIRST_YEAR}`;
    if (num > CURRENT_YEAR) return `Max year is ${CURRENT_YEAR}`;
    return undefined;
}

function validateScore(value: string): string | undefined {
    if (!value) return undefined;
    const num = parseFloat(value);
    if (isNaN(num)) return "Must be a number";
    if (num < 0) return "Min score is 0";
    if (num > 10) return "Max score is 10";
    return undefined;
}

// --- 3. ГЛАВНЫЙ КОМПОНЕНТ ---

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

    const [selectedTypes, setSelectedTypes] = useState<string[]>([]);
    const [sortBy, setSortBy] = useState("relevance");
    const [includeAdult, setIncludeAdult] = useState(false);

    const [errors, setErrors] = useState<ValidationErrors>({});
    const [touched, setTouched] = useState<Record<string, boolean>>({});

    // --- Валидация ---
    const validate = useCallback((): ValidationErrors => {
        const errs: ValidationErrors = {};

        const yearMinErr = validateYear(yearMin);
        if (yearMinErr) errs.yearMin = yearMinErr;

        const yearMaxErr = validateYear(yearMax);
        if (yearMaxErr) errs.yearMax = yearMaxErr;

        if (!yearMinErr && !yearMaxErr && yearMin && yearMax) {
            if (parseInt(yearMin) > parseInt(yearMax)) {
                errs.yearRange = "'From' must be ≤ 'To'";
            }
        }

        const scoreErr = validateScore(minScore);
        if (scoreErr) errs.minScore = scoreErr;

        return errs;
    }, [yearMin, yearMax, minScore]);

    useEffect(() => {
        setErrors(validate());
    }, [validate]);

    const hasErrors = Object.keys(errors).length > 0;

    // --- Активные фильтры ---
    const activeFilterCount =
        selectedGenres.length +
        selectedThemes.length +
        selectedTypes.length +
        (yearMin ? 1 : 0) +
        (yearMax ? 1 : 0) +
        (minScore ? 1 : 0) +
        (sortBy !== "relevance" ? 1 : 0) +
        (includeAdult ? 1 : 0);

    const hasActiveFilters = activeFilterCount > 0;

    const handleResetFilters = () => {
        setSelectedGenres([]);
        setSelectedThemes([]);
        setSelectedTypes([]);
        setYearMin("");
        setYearMax("");
        setMinScore("");
        setSortBy("relevance");
        setIncludeAdult(false);
        setTouched({});
        setErrors({});
    };

    const handleBlur = (field: string) => {
        setTouched(prev => ({...prev, [field]: true}));
    };

    const handleScoreBlur = () => {
        handleBlur("minScore");
        if (minScore) {
            const num = parseFloat(minScore);
            if (!isNaN(num)) {
                const clamped = Math.min(10, Math.max(0, num));
                setMinScore(String(parseFloat(clamped.toFixed(1))));
            }
        }
    };

    const handleYearBlur = (field: "yearMin" | "yearMax") => {
        handleBlur(field);
        const val = field === "yearMin" ? yearMin : yearMax;
        if (val) {
            const num = parseInt(val);
            if (!isNaN(num)) {
                const clamped = Math.min(CURRENT_YEAR, Math.max(ANIME_FIRST_YEAR, num));
                if (field === "yearMin") setYearMin(String(clamped));
                else setYearMax(String(clamped));
            }
        }
    };

    useEffect(() => {
        fetch('/api/filters')
            .then(res => res.json())
            .then(data => {
                if (data.genres && data.genres.length > 0) setAvailableGenres(data.genres);
                if (data.themes && data.themes.length > 0) setAvailableThemes(data.themes);
            })
            .catch(err => console.error("FETCH ERROR:", err));
    }, []);

    const handleSearch = async () => {
        setTouched({yearMin: true, yearMax: true, minScore: true});
        const currentErrors = validate();
        if (Object.keys(currentErrors).length > 0) return;

        if (!query.trim() && selectedGenres.length === 0 && selectedThemes.length === 0 && selectedTypes.length === 0 && !yearMin && !yearMax && !minScore) return;

        setLoading(true);
        setHasSearched(false);
        try {
            const response = await fetch('/api/recommend', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    text_query: query,
                    genres: selectedGenres.length > 0 ? selectedGenres : null,
                    themes: selectedThemes.length > 0 ? selectedThemes : null,
                    type: selectedTypes.length > 0 ? selectedTypes : null,
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

    const showError = (field: keyof ValidationErrors) =>
        touched[field] ? errors[field] : undefined;

    const yearRangeError = (touched.yearMin || touched.yearMax) ? errors.yearRange : undefined;

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
                <section className="bg-[#09090b] border border-zinc-800 rounded-[32px] shadow-2xl overflow-hidden">
                    <div className="p-6 md:p-8 space-y-8">
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

                            {/* RELEASE PERIOD с валидацией */}
                            <div className="space-y-2">
                                <label className="text-[11px] font-bold uppercase tracking-wider text-zinc-500 ml-3">
                                    Release Period
                                    <span className="text-zinc-600 normal-case font-normal ml-1">({ANIME_FIRST_YEAR}–{CURRENT_YEAR})</span>
                                </label>
                                <div className="flex items-start gap-2">
                                    <ValidatedInput
                                        type="text"
                                        inputMode="numeric"
                                        placeholder="From"
                                        value={yearMin}
                                        onChange={(e) => {
                                            const val = e.target.value.replace(/\D/g, '').slice(0, 4);
                                            setYearMin(val);
                                        }}
                                        onBlur={() => handleYearBlur("yearMin")}
                                        error={showError("yearMin")}
                                    />
                                    <ValidatedInput
                                        type="text"
                                        inputMode="numeric"
                                        placeholder="To"
                                        value={yearMax}
                                        onChange={(e) => {
                                            const val = e.target.value.replace(/\D/g, '').slice(0, 4);
                                            setYearMax(val);
                                        }}
                                        onBlur={() => handleYearBlur("yearMax")}
                                        error={showError("yearMax")}
                                    />
                                </div>
                                {yearRangeError && !showError("yearMin") && !showError("yearMax") && (
                                    <p className="text-[11px] text-red-400 ml-1 flex items-center gap-1">
                                        <AlertCircle className="h-3 w-3 shrink-0"/> {yearRangeError}
                                    </p>
                                )}
                            </div>

                            <div className="space-y-2">
                                <label className="text-[11px] font-bold uppercase tracking-wider text-zinc-500 ml-3">Format</label>
                                <MultiSelect
                                    data={TYPES_OPTIONS.map(opt => opt.value)}
                                    selected={selectedTypes}
                                    setSelected={setSelectedTypes}
                                    placeholder="Any Format"
                                />
                            </div>
                        </div>

                        <div className="h-px w-full bg-zinc-800"/>

                        <div className="relative flex items-center w-full bg-zinc-950 border border-zinc-800 rounded-2xl focus-within:ring-1 focus-within:ring-zinc-600 transition-all p-2 pl-4">
                            <Search className="h-5 w-5 text-zinc-500 shrink-0 mr-3"/>
                            <Input
                                placeholder="Describe vibe: 'story about a silent hero in a magical forest'..."
                                value={query}
                                onChange={(e) => setQuery(e.target.value)}
                                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                                className="flex-1 bg-transparent border-none text-base h-12 focus-visible:ring-0 placeholder:text-zinc-600 text-zinc-200 px-0"
                            />
                            <div className="flex items-center gap-2 shrink-0 ml-2">
                                {/* SCORE с валидацией */}
                                <div className="hidden sm:flex flex-col items-start gap-0.5">
                                    <div className={cn(
                                        "flex items-center gap-2 px-3 py-2 rounded-xl bg-[#09090b] border transition-colors",
                                        showError("minScore") ? "border-red-500/60" : "border-zinc-800"
                                    )}>
                                        <Star className={cn("h-4 w-4 shrink-0", showError("minScore") ? "text-red-400" : "text-yellow-500 fill-yellow-500")}/>
                                        <input
                                            type="number"
                                            step="0.1"
                                            min="0"
                                            max="10"
                                            placeholder="0.0"
                                            value={minScore}
                                            onChange={(e) => {
                                                const val = e.target.value;
                                                if (val === "" || /^\d{0,2}(\.\d{0,1})?$/.test(val)) {
                                                    setMinScore(val);
                                                }
                                            }}
                                            onBlur={handleScoreBlur}
                                            className={cn("w-10 bg-transparent border-none text-sm font-semibold focus:outline-none placeholder:text-zinc-600", noSpinners, showError("minScore") ? "text-red-400" : "text-zinc-200")}
                                        />
                                    </div>
                                    {showError("minScore") && (
                                        <p className="flex items-center gap-1 text-[10px] text-red-400 ml-1 whitespace-nowrap">
                                            <AlertCircle className="h-2.5 w-2.5 shrink-0"/> {showError("minScore")}
                                        </p>
                                    )}
                                </div>
                                <div className="h-8 w-px bg-zinc-800 mx-2 hidden sm:block"/>
                                <SortDropdown
                                    value={sortBy}
                                    setValue={setSortBy}
                                    includeAdult={includeAdult}
                                    setIncludeAdult={setIncludeAdult}
                                />
                                <Button
                                    onClick={handleSearch}
                                    disabled={loading}
                                    title={hasErrors ? "Fix validation errors before searching" : undefined}
                                    className={cn(
                                        "h-12 px-8 font-bold rounded-xl transition-all active:scale-95 ml-1",
                                        hasErrors && Object.keys(touched).length > 0
                                            ? "bg-zinc-700 text-zinc-400 cursor-not-allowed"
                                            : "bg-white text-black hover:bg-zinc-200"
                                    )}
                                >
                                    {loading ? "..." : "Find"}
                                </Button>
                            </div>
                        </div>
                    </div>

                    {/* ANIMATED RESET FOOTER */}
                    <div
                        className={cn(
                            "overflow-hidden transition-all duration-500 ease-in-out",
                            hasActiveFilters ? "max-h-20 opacity-100" : "max-h-0 opacity-0"
                        )}
                    >
                        <div className="px-6 md:px-8 pb-5">
                            <div className="flex items-center justify-between border border-zinc-800 rounded-2xl px-4 py-2.5 bg-zinc-950">
                                <div className="flex items-center gap-2 flex-wrap">
                                    <span className="text-[11px] font-bold uppercase tracking-wider text-zinc-500">
                                        Active filters:
                                    </span>
                                    <span className="inline-flex items-center justify-center h-5 min-w-5 px-1.5 rounded-full bg-zinc-700 text-[11px] font-bold text-zinc-200">
                                        {activeFilterCount}
                                    </span>
                                    {selectedGenres.map(g => (
                                        <FilterPill key={g} label={g} onRemove={() => setSelectedGenres(prev => prev.filter(i => i !== g))} />
                                    ))}
                                    {selectedThemes.map(t => (
                                        <FilterPill key={t} label={t} onRemove={() => setSelectedThemes(prev => prev.filter(i => i !== t))} />
                                    ))}
                                    {selectedTypes.map(t => (
                                        <FilterPill key={t} label={t} onRemove={() => setSelectedTypes(prev => prev.filter(i => i !== t))} />
                                    ))}
                                    {yearMin && <FilterPill label={`From ${yearMin}`} onRemove={() => setYearMin("")} />}
                                    {yearMax && <FilterPill label={`To ${yearMax}`} onRemove={() => setYearMax("")} />}
                                    {minScore && <FilterPill label={`★ ≥ ${minScore}`} onRemove={() => setMinScore("")} />}
                                    {sortBy !== "relevance" && <FilterPill label={sortBy} onRemove={() => setSortBy("relevance")} />}
                                    {includeAdult && <FilterPill label="18+" onRemove={() => setIncludeAdult(false)} />}
                                </div>
                                <button
                                    onClick={handleResetFilters}
                                    className="shrink-0 flex items-center gap-1.5 ml-4 text-xs font-semibold text-zinc-400 hover:text-white transition-colors group"
                                >
                                    <RotateCcw className="h-3.5 w-3.5 transition-transform duration-300 group-hover:-rotate-180" />
                                    Reset all
                                </button>
                            </div>
                        </div>
                    </div>
                </section>

                {/* ALERT */}
                {!loading && hasSearched && results.length === 0 && (
                    <div className="max-w-2xl mx-auto">
                        <Alert className="bg-[#09090b] border-zinc-800 text-zinc-200">
                            <AlertCircle className="h-5 w-5 stroke-zinc-400"/>
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
                        <AnimeCard key={anime.mal_id} anime={anime}/>
                    ))}
                </div>
            </main>
        </div>
    );
}

// --- VALIDATED INPUT ---

function ValidatedInput({error, className, ...props}: React.InputHTMLAttributes<HTMLInputElement> & {error?: string}) {
    return (
        <div className="flex-1 space-y-1">
            <Input
                {...props}
                className={cn(
                    "bg-zinc-950 border h-11 rounded-2xl placeholder:text-zinc-600 text-zinc-200 px-4 transition-colors",
                    noSpinners,
                    error
                        ? "border-red-500/60 text-red-300 focus-visible:ring-red-800/40"
                        : "border-zinc-800 focus-visible:ring-zinc-700",
                    className
                )}
            />
            {error && (
                <p className="flex items-center gap-1 text-[11px] text-red-400 ml-1">
                    <AlertCircle className="h-3 w-3 shrink-0"/> {error}
                </p>
            )}
        </div>
    );
}

// --- FILTER PILL ---

function FilterPill({label, onRemove}: { label: string; onRemove: () => void }) {
    return (
        <span className="hidden sm:inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-zinc-800 text-zinc-300 text-[11px] font-medium border border-zinc-700">
            {label}
            <button onClick={onRemove} className="hover:text-white transition-colors ml-0.5">
                <X className="h-2.5 w-2.5"/>
            </button>
        </span>
    );
}

// --- MULTI SELECT ---

function MultiSelect({data, selected, setSelected, placeholder}: {
    data: string[],
    selected: string[],
    setSelected: any,
    placeholder: string
}) {
    const [open, setOpen] = useState(false);
    return (
        <Popover open={open} onOpenChange={setOpen}>
            <PopoverTrigger asChild>
                <Button variant="outline"
                        className="w-full justify-between h-11 bg-zinc-950 border-zinc-800 rounded-2xl text-zinc-300 font-normal hover:bg-zinc-900 hover:text-zinc-100 px-4">
                    <span className="truncate">{selected.length > 0 ? `${selected.length} selected` : placeholder}</span>
                    <ChevronsUpDown className="h-4 w-4 opacity-30"/>
                </Button>
            </PopoverTrigger>
            <PopoverContent className="w-[200px] p-0 bg-[#09090b] border-zinc-800 shadow-xl rounded-xl overflow-hidden" align="start">
                <Command className="bg-transparent">
                    <CommandInput placeholder="Search..." className="text-zinc-200"/>
                    <CommandList>
                        <CommandGroup className="max-h-60 overflow-y-auto p-1">
                            {data.map((item) => (
                                <CommandItem
                                    key={item}
                                    onSelect={() => setSelected(selected.includes(item) ? selected.filter(i => i !== item) : [...selected, item])}
                                    className="cursor-pointer text-zinc-400 aria-selected:bg-zinc-100 aria-selected:text-black rounded-md px-2 py-1.5 transition-colors"
                                >
                                    <Check className={cn("mr-2 h-4 w-4", selected.includes(item) ? "opacity-100" : "opacity-0")}/>
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

// --- ANIME CARD ---

function AnimeCard({anime}: { anime: Anime }) {
    return (
        <div className="group perspective h-[420px] cursor-pointer">
            <div className="relative w-full h-full transition-all duration-500 preserve-3d group-hover:rotate-y-180">
                <div className="absolute inset-0 backface-hidden w-full h-full">
                    <Card className="w-full h-full overflow-hidden border-zinc-800 bg-[#09090b] rounded-2xl border p-0 shadow-lg group-hover:shadow-zinc-900/50">
                        <div className="relative w-full h-full">
                            <img src={anime.image_url} alt=""
                                 className="w-full h-full object-cover grayscale-[40%] group-hover:grayscale-0 transition-all duration-500"/>
                            <div className="absolute inset-0 bg-gradient-to-t from-black via-black/50 to-transparent opacity-80"/>
                            <div className="absolute bottom-0 p-5 w-full space-y-1">
                                <h3 className="font-bold text-lg text-white line-clamp-2 leading-tight">{anime.title}</h3>
                                <div className="flex items-center gap-1 bg-white/10 w-fit px-2 py-0.5 rounded text-white font-bold text-xs backdrop-blur-md">
                                    <Star className="h-3 w-3 text-yellow-500 fill-yellow-500"/>
                                    <span>{anime.score || "N/A"}</span>
                                </div>
                            </div>
                        </div>
                    </Card>
                </div>
                <div className="absolute inset-0 backface-hidden rotate-y-180 w-full h-full">
                    <Card className="w-full h-full bg-[#09090b] border-zinc-800 p-4 flex flex-col rounded-2xl border">
                        <h3 className="font-bold text-base text-white leading-tight mb-2 line-clamp-2">{anime.title}</h3>
                        <div className="h-px bg-zinc-800 w-full mb-2"/>
                        <ScrollArea className="flex-1 min-h-0 pr-4">
                            <p className="text-sm leading-relaxed text-zinc-400 italic">{anime.description || "No description provided."}</p>
                        </ScrollArea>
                        <Button asChild className="mt-2 w-full bg-white text-black hover:bg-zinc-200 rounded-xl font-semibold h-10">
                            <a href={`https://myanimelist.net/anime/${anime.mal_id}`} target="_blank" rel="noreferrer">View on MyAnimeList</a>
                        </Button>
                    </Card>
                </div>
            </div>
        </div>
    );
}

// --- SORT DROPDOWN ---

function SortDropdown({value, setValue, includeAdult, setIncludeAdult}: any) {
    return (
        <DropdownMenu>
            <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon"
                        className="h-12 w-12 rounded-xl text-zinc-400 hover:text-white hover:bg-zinc-800">
                    <SlidersHorizontal className="h-5 w-5"/>
                </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent className="w-56 bg-[#09090b] border-zinc-800 shadow-2xl mr-4 rounded-xl p-1" align="end">
                <DropdownMenuLabel className="text-[10px] font-bold uppercase tracking-widest text-zinc-500 px-3 py-2">Sort Results By</DropdownMenuLabel>
                <DropdownMenuSeparator className="bg-zinc-800 mx-1"/>
                <DropdownMenuRadioGroup value={value} onValueChange={setValue}>
                    <DropdownMenuRadioItem value="relevance" className="cursor-pointer text-zinc-400 focus:text-white focus:bg-zinc-900 rounded-md">Relevance</DropdownMenuRadioItem>
                    <DropdownMenuRadioItem value="rating" className="cursor-pointer text-zinc-400 focus:text-white focus:bg-zinc-900 rounded-md">Top Rated</DropdownMenuRadioItem>
                    <DropdownMenuRadioItem value="popularity" className="cursor-pointer text-zinc-400 focus:text-white focus:bg-zinc-900 rounded-md">Most Popular</DropdownMenuRadioItem>
                </DropdownMenuRadioGroup>
                <DropdownMenuSeparator className="bg-zinc-800 mx-1 my-1"/>
                <div
                    className="flex items-center gap-3 p-2 rounded-md hover:bg-zinc-900 cursor-pointer transition-colors group"
                    onClick={(e) => {
                        e.preventDefault();
                        setIncludeAdult(!includeAdult);
                    }}>
                    <Checkbox checked={includeAdult}
                              className="h-4 w-4 border-zinc-600 data-[state=checked]:bg-white data-[state=checked]:border-white data-[state=checked]:text-black transition-all"/>
                    <span className="text-sm text-zinc-400 group-hover:text-white transition-colors">Include 18+ Content</span>
                </div>
            </DropdownMenuContent>
        </DropdownMenu>
    );
}