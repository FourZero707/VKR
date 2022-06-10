import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# Есть много элементов конфигурации, некоторые из них не нужны, обратитесь к официальному документу https://packaging.python.org/guides/distributing-packages-using-setuptools/
setuptools.setup(
    name="neural", # Имя проекта, который будет установлен через pip install neural в будущем и не может быть повторен с другими проектами, иначе загрузка не удастся
    version="2", # Номер версии проекта, решайте сами
    author="", # Автор
    author_email="", # email
    description="сервисная упаковка рпк",  # Описание Проекта
    long_description=long_description, # Загружаем содержимое read_me
    long_description_content_type="text/markdown", # Тип текста описания
    url="",  # Адрес проекта, например адрес github или gitlib
    packages=setuptools.find_packages(),  # Эта функция может помочь вам найти все файлы в пакете, вы можете указать вручную
    classifiers=[  
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy'
    ], # Зависимости проекта, вы также можете указать зависимую версию
)
