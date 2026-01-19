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
// import { useRef, useEffect } from 'react';
// import { Textarea } from "@/components/ui/textarea";

// Данные для фильтров
const filterOptions = {
  genres: [{ value: "action", label: "Action" }, { value: "comedy", label: "Comedy" }, { value: "drama", label: "Drama" }],
  ratings: [{ value: "pg13", label: "PG-13" }, { value: "r", label: "R - 17+" }],
  types: [{ value: "tv", label: "TV Series" }, { value: "movie", label: "Movie" }]
};

interface Anime {
  mal_id: number;
  title: string;
  description: string;
  score?: number;     // Optional[float]
  image_url?: string; // Optional[str]
  status?: string;    // Optional[str]
}

export default function AnimeApp() {
  const [query, setQuery] = useState("");
  const [genre, setGenre] = useState("");
  const [rating, setRating] = useState("");
  const [type, setType] = useState("");

  // ВОТ ЭТОГО НЕ ХВАТАЛО: Состояние для результатов
  const [results, setResults] = useState<Anime[]>([]);
  const [loading, setLoading] = useState(false);

  // useEffect(() => {
  //   const textarea = textareaRef.current;
  //   if (textarea) {
  //     // Сбрасываем высоту, чтобы она пересчиталась правильно при удалении текста
  //     textarea.style.height = "0px";
  //     // Устанавливаем высоту равную высоте контента (scrollHeight)
  //     // Ограничим максимальную высоту, например, 200px, чтобы поиск не занял весь экран
  //     const scrollHeight = textarea.scrollHeight;
  //     textarea.style.height = Math.min(scrollHeight, 200) + "px";
  //   }
  // }, [query]); // Срабатывает каждый раз, когда меняется текст запроса
  //
  // Функция поиска (пока с моковыми данными)
const handleSearch = async () => {
  if (!query) return;
  setLoading(true);

  try {
    const response = await fetch('http://127.0.0.1:8000/recommend', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      // Шлем ТОЛЬКО text_query, как прописано в твоем Pydantic классе
      body: JSON.stringify({
        text_query: query
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      console.error("Ошибка валидации:", errorData.detail);
      return;
    }

    const data = await response.json();

    // Если бэкенд возвращает RecommendationResponse,
    // то список лежит в data.model_response
    setResults(data.model_response);

  } catch (error) {
    console.error("Ошибка сети:", error);
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
            Персонализированная система рекомендаций аниме с языковой моделью.
          </p>
        </div>

        <section className="bg-zinc-900/50 border border-zinc-800 p-6 rounded-2xl backdrop-blur-sm space-y-6 shadow-2xl">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <FilterCombobox title="Genre" options={filterOptions.genres} value={genre} setValue={setGenre} />
            <FilterCombobox title="Rating" options={filterOptions.ratings} value={rating} setValue={setRating} />
            <FilterCombobox title="Type" options={filterOptions.types} value={type} setValue={setType} />
          </div>

          <div className="relative flex items-center group">
            <Search className="absolute left-4 h-5 w-5 text-zinc-500 group-focus-within:text-zinc-200 transition-colors" />
            <Input
              placeholder="Введите описание (например: темное фэнтези с крутым сюжетом)..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="pl-12 h-14 bg-zinc-950/50 border-zinc-800 text-lg rounded-xl focus-visible:ring-zinc-700 transition-all"
            />
            <Button
              onClick={handleSearch}
              disabled={loading}
              className="absolute right-2 h-10 px-6 bg-zinc-50 text-zinc-950 hover:bg-zinc-200 font-bold rounded-lg transition-all"
            >
              {loading ? "..." : "Найти"}
            </Button>
          </div>
        </section>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-8">
          {results.map((anime) => (
            <AnimeCard key={anime.mal_id} anime={anime} />
          ))}
        </div>
      </main>
    </div>
  );
}

interface FilterOption {
  value: string;
  label: string;
}

// 2. Создаем интерфейс для пропсов самого компонента
interface FilterComboboxProps {
  title: string;
  options: FilterOption[]; // массив объектов FilterOption
  value: string;
  setValue: (value: string) => void; // функция, которая принимает строку и ничего не возвращает
}

// 3. Указываем эти типы в параметрах функции
function FilterCombobox({ title, options, value, setValue }: FilterComboboxProps) {
  const [open, setOpen] = useState(false);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          className="w-full justify-between h-11 bg-zinc-900 border-zinc-800 hover:bg-zinc-800 hover:text-zinc-50 transition-all rounded-xl text-zinc-400"
        >
          {value ? options.find((o) => o.value === value)?.label : `Select ${title}`}
          <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      {/* Добавляем класс dark сюда и явно прописываем фон */}
      <PopoverContent className="w-[200px] p-0 bg-zinc-950 border-zinc-800 shadow-2xl overflow-hidden">
        <Command className="bg-zinc-950 text-zinc-200">
          <CommandInput placeholder={`Search ${title}...`} className="h-9 border-none focus:ring-0" />
          <CommandList className="border-t border-zinc-800">
            <CommandEmpty className="py-2 px-4 text-xs text-zinc-500">No results.</CommandEmpty>
            <CommandGroup className="p-1">
              {options.map((opt) => (
                <CommandItem
                  key={opt.value}
                  value={opt.value}
                  onSelect={(v) => {
                    setValue(v === value ? "" : v);
                    setOpen(false);
                  }}
                  // text-zinc-400 — теперь текст всех элементов виден сразу (серый)
                  // aria-selected — стиль для элемента, на который навели или который выбрали
                  className="rounded-lg cursor-pointer flex items-center px-2 py-2 text-sm outline-none
                             text-zinc-400
                             aria-selected:bg-zinc-800 aria-selected:text-zinc-50
                             hover:bg-zinc-800 hover:text-zinc-50 transition-colors duration-200"
                >
                  {/* Галочка тоже должна быть видна только у выбранного */}
                  <Check className={cn(
                    "mr-2 h-4 w-4 text-primary",
                    value === opt.value ? "opacity-100" : "opacity-0"
                  )} />

                  {/* Само название жанра */}
                  <span className="flex-1">{opt.label}</span>
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}

// КОМПОНЕНТ КАРТОЧКИ (тот самый Flip Card)
function AnimeCard({ anime }: { anime: Anime }) {
  const malLink = `https://myanimelist.net/anime/${anime.mal_id}`;

  return (
    // Фиксируем высоту всей группы (например, 450px)
    <div className="group perspective w-full h-[450px] cursor-pointer">
      <div className="relative w-full h-full transition-all duration-700 preserve-3d group-hover:rotate-y-180">

        {/* ЛИЦЕВАЯ СТОРОНА */}
        <div className="absolute inset-0 backface-hidden">
          <Card className="w-full h-full overflow-hidden border-zinc-800 bg-zinc-950 rounded-2xl border-[1px] p-0 shadow-2xl">
            <div className="relative w-full h-full">
              {/* Картинка ВСЕГДА заполняет всё пространство */}
              <img
                src={anime.image_url || ''}
                alt={anime.title}
                className="absolute inset-0 w-full h-full object-cover object-center transition-transform duration-700 group-hover:scale-110"
              />

              {/* Градиент */}
              <div className="absolute inset-0 bg-gradient-to-t from-zinc-950 via-zinc-950/20 to-transparent opacity-90" />

              {/* Текст снизу */}
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

        {/* ОБРАТНАЯ СТОРОНА */}
        <div className="absolute inset-0 backface-hidden rotate-y-180">
          {/* flex-col + justify-between заставляет кнопку «прилипнуть» к низу */}
          <Card className="w-full h-full bg-zinc-900 border-zinc-700 p-6 flex flex-col justify-between shadow-2xl rounded-2xl border-2">
            <div className="space-y-4 overflow-hidden">
              <div className="space-y-1">
                <h3 className="font-bold text-xl text-zinc-50 tracking-tight line-clamp-2 uppercase">
                  {anime.title}
                </h3>
                {anime.status && (
                  <p className="text-[10px] text-zinc-500 font-bold uppercase tracking-widest">
                    Status: {anime.status}
                  </p>
                )}
              </div>

              <div className="h-px bg-zinc-800 w-full" />

              {/* line-clamp-[10] обрежет текст, если он слишком длинный, чтобы кнопка не уплыла */}
              <p className="text-sm text-zinc-400 leading-relaxed line-clamp-[10] font-light italic">
                {anime.description}
              </p>
            </div>

            <Button asChild className="w-full bg-zinc-50 text-zinc-950 hover:bg-zinc-200 rounded-xl font-bold h-12 mt-4 shrink-0 transition-all active:scale-95 shadow-lg shadow-white/5">
              <a href={malLink} target="_blank" rel="noopener noreferrer">
                View on MyAnimeList
              </a>
            </Button>
          </Card>
        </div>

      </div>
    </div>
  );
}