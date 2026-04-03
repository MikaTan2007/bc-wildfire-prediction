"use client";
import dynamic from "next/dynamic";

const WildfireMap = dynamic(() => import ("./components/WildfireMap"), {
  ssr: false
});

export default function Home() {
  return (
    <main className = "page-root">
      <WildfireMap></WildfireMap>
    </main>
  );
}
