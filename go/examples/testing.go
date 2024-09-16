package main

import (
	"errors"
	"sync"
)

type User struct {
	ID   int
	Name string
}

type UserStore struct {
	mu    sync.RWMutex
	users map[int]User
}

// NewUserStore crea un nuovo "database" di utenti
func NewUserStore() *UserStore {
	return &UserStore{
		users: make(map[int]User),
	}
}

// AddUser aggiunge un utente al database
func (s *UserStore) AddUser(u User) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.users[u.ID]; exists {
		return errors.New("user already exists")
	}

	s.users[u.ID] = u
	return nil
}

// GetUser recupera un utente dal database
func (s *UserStore) GetUser(id int) (User, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	user, exists := s.users[id]
	if !exists {
		return User{}, errors.New("user not found")
	}

	return user, nil
}

// DeleteUser rimuove un utente dal database
func (s *UserStore) DeleteUser(id int) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.users[id]; !exists {
		return errors.New("user not found")
	}

	delete(s.users, id)
	return nil
}
