from typing import Callable, Generic, TypeVar, Optional
import os
import sys
import threading
import time
import pickle

T = TypeVar("T")

class Spinner:
	SPINNER = (
		'⡿',
		'⣟',
		'⣯',
		'⣷',
		'⣾',
		'⣽',
		'⣻',
		'⢿'
	)
	def __init__(self, text: str):
		self.text = text
		self.__start_time = time.monotonic()
		self.__done = threading.Event()
		self.__thread = None
	
	def __spinner_loop(self):
		if sys.stdout.isatty():
			i = 0
			while not self.__done.is_set():
				spinner = Spinner.SPINNER[i]
				print(f"\r{self.text} {spinner} ", end="")
				i = (i + 1) % len(Spinner.SPINNER)
				self.__done.wait(0.1)
		duration = round(time.monotonic() - self.__start_time, 3)
		print(f"\r{self.text} in {duration}s")

	def __enter__(self):
		self.__done.clear()
		self.__thread = threading.Thread(target=self.__spinner_loop)
		self.__thread.start()

	def __exit__(self, type, value, traceback):
		self.__done.set()
		self.__thread.join()

class Cache(Generic[T]):
	DIR = "./.cache/"
	def __init__(self, id: str, version: int):
		self.name = id
		self.version = version
		self.__path = os.path.join(Cache.DIR, f"{id}-v{version}.pkl")

	def try_load(self) -> Optional[T]:
		try:
			os.makedirs(Cache.DIR, exist_ok=True)
			with open(self.__path, "rb") as f:
				return pickle.load(f)
		except FileNotFoundError:
			return None


	def save(self, value: T):
		with open(self.__path, "wb") as f:
			pickle.dump(value, f)


def cached(id: str, version: int, fn: Callable[[], T]) -> T:
	cache = Cache(id, version)
	value = cache.try_load()
	if value is None:
		value = fn()
		cache.save(value)
	return value
