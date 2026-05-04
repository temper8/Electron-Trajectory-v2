import os
from pathlib import Path

from src.config import read_config_hdf5

import os
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, IntPrompt
from rich.panel import Panel

def select_h5_file(directory="race"):
    console = Console()
    base_path = Path(directory)

    if not base_path.exists():
        console.print(f"[bold red]✘ Папка '{directory}' не найдена.[/bold red]")
        return None

    # Поиск 5 самых новых файлов
    files = sorted(
        [f for f in base_path.rglob("*.h5") if f.is_file()],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )[:5]

    if not files:
        console.print("[yellow]! Файлы .h5 не найдены.[/yellow]")
        return None

    console.print(f"\n[bold cyan]Последние сессии в {directory}:[/bold cyan]\n")
    
    for i, file in enumerate(files, 1):
        mtime = datetime.fromtimestamp(file.stat().st_mtime).strftime('.%m.%d %H:%M')
        
        # Печатаем заголовок файла
        console.print(f"[bold green]{i}[/bold green]) [bold white]{file.name}[/bold white] [dim]({mtime})[/dim]")
        
        # Читаем и выводим параметры файла
        try:
            solver, params = read_config_hdf5(str(file))
            console.print(f"   [bold blue]Solver:[/bold blue] {solver}")
            console.print(f"   [bold blue]Params:[/bold blue] {params}")
        except Exception as e:
            console.print(f"   [red]Ошибка чтения параметров: {e}[/red]")
        
        console.print("") # Пустая строка для разделения блоков

    choice = IntPrompt.ask(
        "Выберите файл [dim](0 для выхода)[/dim]", 
        choices=[str(i) for i in range(len(files) + 1)],
        show_choices=False
    )

    if choice == 0:
        return None

    selected = files[choice - 1]
    console.print(f"\n[bold green]✔ Выбрано:[/bold green] {selected.name}")
    
    return selected

if __name__ == "__main__":
    selected_path = select_h5_file()

