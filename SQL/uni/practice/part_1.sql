### Parte 1
1. Categorie assegnate a più di un film.
2. E i nomi delle categorie? (eventualmente ordinati).
3. Nomi attori che hanno recitato in film Comedy e che hanno recitato insieme a qualche attore di nome 'Ralph'.
4. Attori che non hanno mai recitato in un film che dura meno di 60 minuti.
5. Per ogni categoria di film, la durata totale di tutti i film di quella categoria.
6. E per nome di categoria, e ordinati dai più lunghi ai più corti?
7. Come la precedente interrogazione, ma voglio solo le categorie che hanno totalizzato più di 7000 minuti in totale.
8. I film che durano di più.
9. I film in cui hanno recitato tutti gli attori di nome 'Spencer'.

select distinct f1.category_id
from film_category as f1, film_category as f2
where f1.film_id <> f2.film_id and f1.category_id = f2.category_id;

select distinct c.name
from film_category as f1, film_category as f2, category as c
where f1.film_id <> f2.film_id and f1.category_id = f2.category_id and f1.category_id = c.category_id
order by c.name;


