-- Definizione dell'entità Adder4
entity Adder4 is
  port (
    A, B    : in bit_vector(3 downto 0);
    Cin     : in bit;
    
    -- Uscite: un vettore a 4 bit e un bit singolo per il carry out
    S       : out bit_vector(3 downto 0);
    Cout    : out bit
  );
end Adder4;

-- Architettura strutturale dell'Adder4
architecture Structural of Adder4 is
  signal C : bit_vector(3 downto 1); -- Segnali interni per i carry

begin
  -- Istanziazione dei FullAdder con mappatura posizionale
  FA0: entity work.FullAdder
    port map (
      A    => A(0),
      B    => B(0),
      Cin  => Cin,
      Sum  => S(0),
      Cout => C(1)
    );

  FA1: entity work.FullAdder
    port map (
      A    => A(1),
      B    => B(1),
      Cin  => C(1),
      Sum  => S(1),
      Cout => C(2)
    );

  FA2: entity work.FullAdder
    port map (
      A    => A(2),
      B    => B(2),
      Cin  => C(2),
      Sum  => S(2),
      Cout => C(3)
    );

  FA3: entity work.FullAdder
    port map (
      A    => A(3),
      B    => B(3),
      Cin  => C(3),
      Sum  => S(3),
      Cout => Cout
    );

end architecture Structural;

-- Definizione dell'entità TestAdder4
entity TestAdder4 is
end TestAdder4;

-- Architettura del testbench
architecture Behavioral of TestAdder4 is
  -- Segnali per collegare al DUT (Device Under Test)
  signal A, B : bit_vector(3 downto 0);
  signal Cin : bit;
  signal S, Cout : bit_vector(3 downto 0); -- Nota: Cout è un singolo bit, ma per coerenza lo manteniamo come vettore

begin
  -- Istanziazione del DUT: Adder4
  dut: entity work.Adder4(Structural)
    port map (
      A    => A,
      B    => B,
      Cin  => Cin,
      S    => S,
      Cout => Cout(0) -- Collegamento del carry out
    );

  -- Generazione degli stimoli
  stim_proc: process
  begin
    -- Inizializzazione
    A <= "0000";
    B <= "0000";
    Cin <= '0';
    wait for 50 ns;
    
    -- Primo stimolo
    A <= "0010";
    B <= "1010";
    Cin <= '1';
    wait for 100 ns;
    
    -- Secondo stimolo
    A <= "1110";
    B <= "1110";
    Cin <= '0';
    wait for 100 ns;
    
    -- Terzo stimolo
    A <= "0110";
    B <= "0110";
    Cin <= '1';
    wait for 100 ns;
    
    -- Fine della simulazione
    wait;
  end process stim_proc;

end architecture Behavioral;

