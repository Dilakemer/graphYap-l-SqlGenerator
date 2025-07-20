#!/usr/bin/env python3
# turkish_nlp_sql_cli.py
"""
Turkish NLP-SQL Terminal CLI Tool
Interactive command-line interface for natural language to SQL conversion
"""

import sys
import os
from pathlib import Path
import time
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src" / "nlp"))
sys.path.append(str(Path(__file__).parent / "src" / "query_builder"))

# Import components
from nlp_processor import NLPProcessor
from sql_generator import SQLGenerator


class TurkishNLPSQLCLI:
    """
    Interactive CLI for Turkish NLP to SQL conversion
    """

    def __init__(self):
        self.nlp_processor = None
        self.sql_generator = None
        self.is_initialized = False
        self.session_queries = 0
        self.successful_queries = 0

        # CLI Configuration
        self.show_debug = False
        self.show_confidence = True
        self.auto_execute = False  # Future: DB execution

    def initialize(self):
        """Initialize NLP and SQL components"""
        print("🚀 Turkish NLP-SQL System Başlatılıyor...")
        print("=" * 50)

        try:
            print("📦 NLP İşlemci yükleniyor...")
            self.nlp_processor = NLPProcessor()

            print("🔧 SQL Üretici yükleniyor...")
            self.sql_generator = SQLGenerator()

            self.is_initialized = True
            print("✅ Sistem başarıyla yüklendi!")

            # Show system info
            self.show_system_info()

            return True

        except Exception as e:
            print(f"❌ Sistem başlatılamadı: {e}")
            return False

    def show_system_info(self):
        """Display system information"""
        if not self.is_initialized:
            return

        print(f"\n📊 Sistem Bilgileri:")
        print("-" * 30)

        # NLP Stats
        nlp_stats = self.nlp_processor.get_processing_stats()
        print(f"🤖 NLP Modeli: {nlp_stats['extraction_method']}")
        print(f"🎯 Desteklenen Intentler: SELECT, COUNT, SUM, AVG")

        # SQL Stats
        sql_stats = self.sql_generator.get_supported_features()
        print(f"📊 Desteklenen Tablolar: {len(sql_stats['supported_tables'])}")
        print(f"⏰ Zaman Filtreleri: {len(sql_stats['time_filters'])}")

        print(f"💾 Model Durumu: Eğitilmiş NER modeli aktif")
        print(f"🔒 Güvenlik: SQL injection koruması aktif")

    def show_help(self):
        """Show help information"""
        print(f"\n💡 Kullanım Kılavuzu:")
        print("-" * 30)
        print(f"📝 Türkçe sorular sorun, sistem SQL'e çevirecek!")
        print(f"")
        print(f"🎯 Örnek Sorgular:")
        print(f"  • Bu ayın müşteri sayısı")
        print(f"  • Geçen yıl sipariş toplamı")
        print(f"  • Son 30 gün ürün listesi")
        print(f"  • Kategori sayısını göster")
        print(f"  • Çalışan maaş ortalaması")
        print(f"")
        print(f"🔧 Komutlar:")
        print(f"  • help, yardım     - Bu yardım menüsünü göster")
        print(f"  • info, bilgi      - Sistem bilgilerini göster")
        print(f"  • debug on/off     - Debug modunu aç/kapat")
        print(f"  • confidence on/off- Güven skorlarını göster/gizle")
        print(f"  • stats, istatistik- Oturum istatistikleri")
        print(f"  • clear, temizle   - Ekranı temizle")
        print(f"  • exit, çıkış      - Programdan çık")

    def process_query(self, user_input):
        """Process user natural language query"""
        if not user_input.strip():
            return

        self.session_queries += 1
        start_time = time.time()

        print(f"\n🔍 Sorgu: '{user_input}'")
        print("-" * 40)

        try:
            # Step 1: NLP Analysis
            print("📝 1️⃣  NLP Analizi...")
            nlp_result = self.nlp_processor.analyze(user_input)

            if self.show_confidence:
                confidence = nlp_result['intent']['confidence']
                print(f"  Intent: {nlp_result['intent']['type']} (güven: {confidence:.1%})")
                print(f"  Tablolar: {len(nlp_result['entities']['tables'])}")
                print(f"  Zaman filtreleri: {len(nlp_result['entities']['time_filters'])}")

            if self.show_debug:
                print(f"  🔍 Debug: {nlp_result['analysis_metadata']}")

            # Step 2: SQL Generation
            print("🔧 2️⃣  SQL Üretimi...")
            sql_result = self.sql_generator.generate_sql(nlp_result)

            if sql_result["success"]:
                print(f"✅ 3️⃣  SQL Başarıyla Üretildi!")
                print(f"")
                print(f"📋 SQL Sorgusu:")
                print(f"```sql")
                print(f"{sql_result['sql']}")
                print(f"```")
                print(f"")
                print(f"📊 Detaylar:")
                print(f"  • Tablo: {sql_result['table']}")
                print(f"  • İşlem: {sql_result['intent']}")
                print(f"  • Zaman filtresi: {'Evet' if sql_result['has_time_filter'] else 'Hayır'}")
                print(f"  • Güven skoru: {sql_result['confidence']:.1%}")

                self.successful_queries += 1

            else:
                print(f"❌ 3️⃣  SQL Üretilemedi!")
                print(f"  Hata: {sql_result['error']}")

                if 'debug_info' in sql_result and self.show_debug:
                    print(f"  Debug: {sql_result['debug_info']}")

                # Suggestions
                self.suggest_improvements(nlp_result, sql_result)

            # Show timing
            elapsed = time.time() - start_time
            print(f"⏱️  İşlem süresi: {elapsed:.2f} saniye")

        except Exception as e:
            print(f"💥 Beklenmeyen Hata: {e}")
            if self.show_debug:
                import traceback
                traceback.print_exc()

    def suggest_improvements(self, nlp_result, sql_result):
        """Suggest improvements for failed queries"""
        print(f"\n💡 Öneriler:")

        # Check intent confidence
        intent_conf = nlp_result['intent']['confidence']
        if intent_conf < 0.7:
            print(f"  • İstediğiniz işlemi daha net belirtin (göster, say, topla, ortala)")

        # Check table detection
        tables = nlp_result['entities']['tables']
        if not tables:
            print(f"  • Hangi tablo üzerinde işlem yapmak istediğinizi belirtin")
            print(f"    (müşteri, ürün, sipariş, kategori, tedarikçi, çalışan)")

        # Check overall structure
        if not nlp_result['analysis_metadata']['sql_ready']:
            print(f"  • Daha basit bir cümle kurmayı deneyin")
            print(f"  • Örnek: 'müşteri sayısı' veya 'bu ayın sipariş toplamı'")

    def handle_command(self, command):
        """Handle special commands"""
        cmd = command.lower().strip()

        if cmd in ['help', 'yardım']:
            self.show_help()

        elif cmd in ['info', 'bilgi']:
            self.show_system_info()

        elif cmd.startswith('debug'):
            if 'on' in cmd or 'aç' in cmd:
                self.show_debug = True
                print("✅ Debug modu açıldı")
            elif 'off' in cmd or 'kapat' in cmd:
                self.show_debug = False
                print("✅ Debug modu kapatıldı")
            else:
                print(f"Debug modu: {'Açık' if self.show_debug else 'Kapalı'}")

        elif cmd.startswith('confidence') or cmd.startswith('güven'):
            if 'on' in cmd or 'aç' in cmd:
                self.show_confidence = True
                print("✅ Güven skorları gösteriliyor")
            elif 'off' in cmd or 'kapat' in cmd:
                self.show_confidence = False
                print("✅ Güven skorları gizlendi")
            else:
                print(f"Güven skorları: {'Gösteriliyor' if self.show_confidence else 'Gizli'}")

        elif cmd in ['stats', 'istatistik']:
            self.show_session_stats()

        elif cmd in ['clear', 'temizle']:
            os.system('cls' if os.name == 'nt' else 'clear')
            self.show_banner()

        else:
            print("❓ Bilinmeyen komut. 'help' yazarak yardım alabilirsiniz.")

    def show_session_stats(self):
        """Show session statistics"""
        success_rate = (self.successful_queries / self.session_queries * 100) if self.session_queries > 0 else 0

        print(f"\n📈 Oturum İstatistikleri:")
        print("-" * 25)
        print(f"📊 Toplam sorgu: {self.session_queries}")
        print(f"✅ Başarılı: {self.successful_queries}")
        print(f"❌ Başarısız: {self.session_queries - self.successful_queries}")
        print(f"🎯 Başarı oranı: {success_rate:.1f}%")

        # System stats
        if self.nlp_processor:
            nlp_stats = self.nlp_processor.get_processing_stats()
            sql_stats = self.sql_generator.get_statistics()

            print(f"\n🤖 Sistem İstatistikleri:")
            print(f"  NLP başarı oranı: {nlp_stats['success_rate']:.1f}%")
            print(f"  SQL başarı oranı: {sql_stats['success_rate']:.1f}%")

    def show_banner(self):
        """Show welcome banner"""
        print(f"""
╔══════════════════════════════════════════════════════════╗
║              🇹🇷 Turkish NLP-SQL System 🇹🇷              ║
║                                                          ║
║    Türkçe doğal dil ile veritabanı sorgulama sistemi    ║
║          Natural Language to SQL Converter               ║
╚══════════════════════════════════════════════════════════╝

💡 'help' yazarak başlayabilir, 'çıkış' ile ayrılabilirsiniz
⚡ Hazır! Türkçe sorgunuzu yazın...
""")

    def run(self):
        """Main CLI loop"""
        # Initialize system
        if not self.initialize():
            return

        # Show banner
        self.show_banner()

        # Main loop
        try:
            while True:
                try:
                    # Get user input
                    user_input = input("\n🔮 Turkish NLP-SQL> ").strip()

                    if not user_input:
                        continue

                    # Check for exit commands
                    if user_input.lower() in ['exit', 'quit', 'çıkış', 'exit()', 'q']:
                        break

                    # Check for special commands
                    if user_input.startswith(
                            ('help', 'info', 'debug', 'confidence', 'güven', 'stats', 'istatistik', 'clear',
                             'temizle')):
                        self.handle_command(user_input)
                    else:
                        # Process as NLP query
                        self.process_query(user_input)

                except KeyboardInterrupt:
                    print(f"\n\n⚠️  İşlem iptal edildi. Çıkmak için 'çıkış' yazın.")
                    continue

                except EOFError:
                    break

        except Exception as e:
            print(f"\n💥 Kritik hata: {e}")

        finally:
            # Show goodbye message
            print(f"\n👋 Güle güle! Turkish NLP-SQL sistemi kapanıyor...")
            if self.session_queries > 0:
                self.show_session_stats()
            print(f"🙏 Sistem kullanımınız için teşekkürler!")


def main():
    """Entry point"""
    cli = TurkishNLPSQLCLI()
    cli.run()


if __name__ == "__main__":
    main()